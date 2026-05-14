# -*- coding: utf-8 -*-
"""
大模型应用开发辅助工具集（精简版）
功能：Prompt优化、数据清洗与拆分、AI数据质量增强与报告、技术文档生成、API成本监控
"""
import os
import re
import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from dashscope import Generation
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import pyperclip

# ====================== 全局配置 ======================
@dataclass(frozen=True)
class AppConfig:
    DEFAULT_MODEL: str = "qwen-turbo"
    SERVER_HOST: str = "localhost"
    SERVER_PORT: int = 7860
    OUTPUT_DIR: str = "./output"
    CHART_DIR: str = "./output/charts"
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    PRICE_PER_KTOKEN: float = 0.008

config = AppConfig()

# ====================== 日志与工具 ======================
def setup_logger():
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(config.OUTPUT_DIR) / "app.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def save_dataframe(df, filename):
    path = Path(config.OUTPUT_DIR) / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)

def validate_csv_file(file):
    if not file:
        raise ValueError("请先上传CSV文件")
    for enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
        try:
            df = pd.read_csv(file.name, encoding=enc)
            if df.empty:
                raise ValueError("CSV文件为空")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"读取失败：{e}")
    raise ValueError("无法识别编码，请另存为UTF-8")

# ====================== 成本追踪器 ======================
class CostTracker:
    _records = []
    _budget = None

    @classmethod
    def add_record(cls, model, input_tokens, output_tokens):
        cls._records.append({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        })

    @classmethod
    def get_stats(cls):
        total_in = sum(r['input_tokens'] for r in cls._records)
        total_out = sum(r['output_tokens'] for r in cls._records)
        total = total_in + total_out
        cost = total * config.PRICE_PER_KTOKEN / 1000
        return {
            'total_input_tokens': total_in,
            'total_output_tokens': total_out,
            'total_tokens': total,
            'estimated_cost_yuan': round(cost, 6),
            'num_calls': len(cls._records)
        }

    @classmethod
    def plot_trend(cls):
        if not cls._records:
            return None
        sorted_records = sorted(cls._records, key=lambda x: x['timestamp'])
        timestamps = [r['timestamp'] for r in sorted_records]
        cumulative = []
        total = 0
        for r in sorted_records:
            total += r['input_tokens'] + r['output_tokens']
            cumulative.append(total)
        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, cumulative, marker='o')
        plt.title('累计Token消耗趋势')
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = Path(config.OUTPUT_DIR) / "cost_trend.png"
        plt.savefig(path)
        plt.close()
        return str(path)

    @classmethod
    def set_budget(cls, tokens):
        cls._budget = tokens

    @classmethod
    def get_budget(cls):
        return cls._budget

    @classmethod
    def get_remaining(cls):
        if cls._budget is not None:
            used = cls.get_stats()['total_tokens']
            return cls._budget - used
        return None

# ====================== 大模型调用服务 ======================
class ModelService:
    _api_keys = {}

    @classmethod
    def set_api_key(cls, model, key):
        cls._api_keys[model] = key

    @classmethod
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=config.RETRY_DELAY, max=5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def call(cls, prompt, model=config.DEFAULT_MODEL, temperature=0.7):
        if model == "qwen-turbo":
            key = cls._api_keys.get("qwen-turbo", os.getenv("DASHSCOPE_API_KEY"))
            if not key:
                raise ValueError("API密钥未配置，请在“配置与监控”页签设置")
            resp = Generation.call(
                model="qwen-turbo",
                api_key=key,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                result_format="message",
                timeout=30
            )
            if resp.status_code != 200:
                raise Exception(f"API调用失败：{resp.message}")
            if hasattr(resp, 'usage'):
                CostTracker.add_record(model, resp.usage.input_tokens, resp.usage.output_tokens)
            return resp.output.choices[0].message.content.strip()
        else:
            raise ValueError(f"不支持的模型：{model}")

# ====================== 业务服务 ======================
class PromptService:
    @staticmethod
    def optimize(original, scene, context=""):
        if not original.strip():
            return "", "请输入原始Prompt"
        system = f"""
        你是专业的Prompt优化工程师，你要给用户的原始Prompt优化，优化要求：
         1. 明确角色定位
         2. 清晰说明任务目标
         3. 添加具体的约束条件
         4. 规定输出格式
         5. 补充必要的背景信息
         6. 语言简洁专业，逻辑、结构清晰。
        场景：{scene}
        附加上下文：{context if context else "无"}
        请严格按下面格式输出：
        ===优化后的Prompt===
        [优化后的内容]
        ===优化说明===
        [优化点说明]
        """
        result = ModelService.call(system + "\n原始Prompt：" + original)
        parts = result.split("===优化说明===")
        optimized = parts[0].replace("===优化后的Prompt===", "").strip()
        reason = parts[1].strip() if len(parts) > 1 else "无"
        return optimized, reason

class DataService:
    @staticmethod
    def clean_csv(file):
        try:
            df = validate_csv_file(file)
            orig = len(df)
            df = df.dropna(how='all').fillna('').drop_duplicates()
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.replace(r'[^\w\s\u4e00-\u9fa5]', '', regex=True)
            cleaned = len(df)
            path = save_dataframe(df, "cleaned_data.csv")
            return f"清洗完成！原始 {orig} 行 → 清洗后 {cleaned} 行\n文件已保存至 {path}"
        except Exception as e:
            return f"清洗失败：{e}"

    @staticmethod
    def split_dataset(file, train, val, test):
        try:
            if abs(train + val + test - 1.0) > 1e-6:
                raise ValueError("三个比例之和必须等于1")
            df = validate_csv_file(file)
            total = len(df)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            t_end = int(total * train)
            v_end = t_end + int(total * val)
            save_dataframe(df[:t_end], "train.csv")
            save_dataframe(df[t_end:v_end], "val.csv")
            save_dataframe(df[v_end:], "test.csv")
            return f"拆分完成！训练集 {t_end} 条，验证集 {v_end - t_end} 条，测试集 {total - v_end} 条"
        except Exception as e:
            return f"拆分失败：{e}"

class DataQualityService:
    @staticmethod
    def evaluate_and_enhance(file):
        df = validate_csv_file(file)
        orig = len(df)
        df = df.dropna(how='all').fillna('').drop_duplicates()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.replace(r'[^\w\s\u4e00-\u9fa5]', '', regex=True)
        cleaned = len(df)

        summary = df.describe(include='all').to_string()
        missing = df.isnull().sum().to_string()
        quality = ModelService.call(
            f"仅回答'优'或'差'。\n统计摘要：{summary}\n缺失值：{missing}\n行数：{cleaned}",
            temperature=0
        ).strip()
        is_good = '优' in quality
        status = ""
        if not is_good:
            sample = df.head(100).to_json(orient='records', force_ascii=False)
            enhanced = ModelService.call(f"清洗以下数据中的错误，直接返回JSON数组，不要解释：{sample}", temperature=0)
            try:
                enhanced_df = pd.DataFrame(json.loads(enhanced))
                if len(enhanced_df) == len(sample):
                    df.iloc[:len(enhanced_df)] = enhanced_df.values
                status = "质量评估：差，已通过AI增强处理。"
            except:
                status = "质量评估：差，但AI增强解析失败，请检查数据格式。"
        else:
            status = "质量评估：优。"

        path = save_dataframe(df, "final_cleaned_data.csv")
        status += f"\n清洗后行数：{cleaned}，文件已保存至 {path}"

        report = ModelService.call(
            f"根据以下信息生成200字以内的数据质量分析报告：\n原始行数 {orig}，清洗后行数 {cleaned}\n字段：{list(df.columns)}\n缺失值统计：{missing}",
            temperature=0.3
        )

        # 生成图表（中文支持）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        chart_dir = Path(config.CHART_DIR)
        chart_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        missing_vals = df.isnull().sum()
        missing_vals = missing_vals[missing_vals > 0]
        if not missing_vals.empty:
            missing_vals.plot(kind='bar', color='#3B82F6')
            plt.title('缺失值统计')
            plt.tight_layout()
            p = chart_dir / "missing.png"
            plt.savefig(p); plt.close()
            paths.append(str(p))

        text_cols = df.select_dtypes(include='object').columns
        if len(text_cols) > 0:
            col = text_cols[0]
            lengths = df[col].astype(str).apply(len)
            plt.figure()
            lengths.hist(bins=20, color='#10B981')
            plt.title(f'“{col}”文本长度分布')
            plt.tight_layout()
            p = chart_dir / "text_len.png"
            plt.savefig(p); plt.close()
            paths.append(str(p))

        for col in df.select_dtypes(include='object').columns[:2]:
            if df[col].nunique() <= 10:
                df[col].value_counts().plot(kind='bar', color='#F59E0B')
                plt.title(f'“{col}”值分布')
                plt.tight_layout()
                p = chart_dir / f"{col}_dist.png"
                plt.savefig(p); plt.close()
                paths.append(str(p))

        return status, report, paths

class DocumentService:
    @staticmethod
    def generate_readme(name, desc, stack, author):
        if not all([name, desc, stack, author]):
            return "请填写所有必填项"
        prompt = f"生成一个标准的GitHub README文档（Markdown格式），包含项目简介、功能特点、技术栈、环境搭建、使用说明、作者信息。项目名：{name}，描述：{desc}，技术栈：{stack}，作者：{author}"
        readme = ModelService.call(prompt)
        path = Path(config.OUTPUT_DIR) / "README.md"
        path.write_text(readme, encoding="utf-8")
        return readme

# ====================== UI界面 ======================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --primary: #6366f1;
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --text-secondary: #94a3b8;
    --border: #334155;
    --radius: 12px;
}
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh;
}
.gradio-tabs, .gradio-accordion {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.tab-nav button {
    background: var(--card) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
}
.tab-nav button.selected {
    background: var(--primary) !important;
    color: white !important;
}
.primary-btn {
    background: linear-gradient(135deg, var(--primary), #4f46e5) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
@media (prefers-color-scheme: light) {
    .gradio-container {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    .gradio-tabs, .gradio-accordion, .gradio-row, .gradio-group {
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #0f172a !important;
    }
    label, .gradio-markdown {
        color: #0f172a !important;
    }
    .gradio-textbox textarea, .gradio-textbox input {
        background: #ffffff !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }
}
"""

def create_ui():
    with gr.Blocks(title="大模型应用开发辅助工具集") as demo:
        gr.Markdown("# 大模型应用开发辅助工具集")
        gr.Markdown("覆盖Prompt优化、数据预处理、AI质量增强、文档生成与成本监控的一站式效率工具")

        with gr.Tabs():
            # ---------- 1. Prompt优化 ----------
            with gr.TabItem("Prompt优化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        original_input = gr.Textbox(label="原始Prompt", lines=6, placeholder="请输入需要优化的提示词")
                        scene_select = gr.Dropdown(label="使用场景", choices=["通用问答", "代码编写", "数据分析", "模型微调", "文本生成"], value="通用问答")
                        context_input = gr.Textbox(label="附加上下文（可选）", lines=2, placeholder="例如：我正在开发一个电商项目...")
                        optimize_btn = gr.Button("一键优化Prompt", elem_classes="primary-btn")
                    with gr.Column(scale=1):
                        optimized_output = gr.Textbox(label="优化后的Prompt", lines=6)
                        copy_btn = gr.Button("复制到剪贴板", elem_classes="primary-btn")
                        copy_status = gr.Textbox(visible=False)
                reason_output = gr.Textbox(label="优化说明", lines=3)

                def copy_text(text):
                    if not text:
                        return "无内容可复制"
                    pyperclip.copy(text)
                    return f"已复制 ({time.strftime('%H:%M:%S')})"

                optimize_btn.click(
                    fn=PromptService.optimize,
                    inputs=[original_input, scene_select, context_input],
                    outputs=[optimized_output, reason_output]
                )
                copy_btn.click(fn=copy_text, inputs=optimized_output, outputs=copy_status)

            # ---------- 2. 数据工作台 ----------
            with gr.TabItem("数据工作台"):
                with gr.Tabs():
                    with gr.TabItem("清洗与拆分"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 数据清洗")
                                clean_file = gr.File(label="上传CSV", file_types=[".csv"])
                                clean_btn = gr.Button("一键清洗", elem_classes="primary-btn")
                                clean_result = gr.Textbox(label="清洗结果", lines=4)
                            with gr.Column():
                                gr.Markdown("### 数据集拆分")
                                split_file = gr.File(label="上传CSV", file_types=[".csv"])
                                train_ratio = gr.Slider(label="训练集比例", minimum=0.5, maximum=0.9, value=0.7, step=0.05)
                                val_ratio = gr.Slider(label="验证集比例", minimum=0.05, maximum=0.3, value=0.2, step=0.05)
                                test_ratio = gr.Slider(label="测试集比例", minimum=0.05, maximum=0.2, value=0.1, step=0.05)
                                split_btn = gr.Button("拆分数据集", elem_classes="primary-btn")
                                split_result = gr.Textbox(label="拆分结果", lines=4)
                        clean_btn.click(fn=DataService.clean_csv, inputs=clean_file, outputs=clean_result)
                        split_btn.click(fn=DataService.split_dataset, inputs=[split_file, train_ratio, val_ratio, test_ratio], outputs=split_result)

                    with gr.TabItem("AI质量增强"):
                        quality_file = gr.File(label="上传CSV", file_types=[".csv"])
                        quality_btn = gr.Button("开始AI评估与增强", elem_classes="primary-btn")
                        quality_status = gr.Textbox(label="处理状态", lines=2)
                        quality_report = gr.Textbox(label="AI质量报告", lines=4)
                        with gr.Row():
                            chart1 = gr.Image(type="filepath", visible=False)
                            chart2 = gr.Image(type="filepath", visible=False)
                            chart3 = gr.Image(type="filepath", visible=False)
                            chart4 = gr.Image(type="filepath", visible=False)

                        def run_quality(file):
                            status, report, paths = DataQualityService.evaluate_and_enhance(file)
                            imgs = []
                            for i in range(4):
                                if i < len(paths):
                                    imgs.append(gr.update(value=paths[i], visible=True))
                                else:
                                    imgs.append(gr.update(visible=False))
                            return [status, report] + imgs

                        quality_btn.click(
                            fn=run_quality,
                            inputs=quality_file,
                            outputs=[quality_status, quality_report, chart1, chart2, chart3, chart4]
                        )

            # ---------- 3. 技术文档生成 ----------
            with gr.TabItem("技术文档生成"):
                with gr.Row():
                    with gr.Column():
                        name = gr.Textbox(label="项目名称", value="大模型应用开发辅助工具集")
                        desc = gr.Textbox(label="项目描述", lines=3, value="集成Prompt优化、数据预处理、AI质量增强与成本监控的工具集")
                        stack = gr.Textbox(label="技术栈", value="Python, Gradio, 通义千问, Pandas, Matplotlib")
                        author = gr.Textbox(label="作者", value="开发者")
                        doc_btn = gr.Button("生成README", elem_classes="primary-btn")
                    with gr.Column():
                        readme_out = gr.Textbox(label="生成的README", lines=12)
                doc_btn.click(fn=DocumentService.generate_readme, inputs=[name, desc, stack, author], outputs=readme_out)

            # ---------- 4. 配置与监控 ----------
            with gr.TabItem("配置与监控"):
                with gr.Accordion("API密钥", open=False):
                    key_input = gr.Textbox(label="API Key", type="password", placeholder="输入通义千问API Key")
                    key_btn = gr.Button("保存密钥", elem_classes="primary-btn")
                    key_status = gr.Textbox(label="状态", interactive=False)
                    key_btn.click(
                        fn=lambda key: (ModelService.set_api_key("qwen-turbo", key), "密钥已保存")[1],
                        inputs=key_input, outputs=key_status
                    )
                with gr.Accordion("成本监控", open=True):
                    budget_input = gr.Number(label="总预算 (tokens)", value=100000)
                    budget_btn = gr.Button("保存预算", elem_classes="primary-btn")
                    refresh_btn = gr.Button("刷新统计", elem_classes="primary-btn")
                    cost_stats = gr.JSON(label="累计用量统计")
                    cost_trend = gr.Image(type="filepath", label="Token消耗趋势")
                    remaining_text = gr.Textbox(label="剩余Token", interactive=False)

                    def refresh_cost():
                        stats = CostTracker.get_stats()
                        trend = CostTracker.plot_trend()
                        remain = CostTracker.get_remaining()
                        budget = CostTracker.get_budget()
                        if remain is not None and budget is not None:
                            remain_str = f"已用：{stats['total_tokens']:,} / 预算：{budget:,} / 剩余：{remain:,}"
                        else:
                            remain_str = "未设定预算"
                        return stats, trend if trend else None, remain_str

                    budget_btn.click(
                        fn=lambda val: (CostTracker.set_budget(int(val)), refresh_cost())[1],
                        inputs=budget_input, outputs=[cost_stats, cost_trend, remaining_text]
                    )
                    refresh_btn.click(fn=refresh_cost, outputs=[cost_stats, cost_trend, remaining_text])

    return demo

if __name__ == "__main__":
    logger.info("应用启动中...")
    demo = create_ui()
    demo.launch(
        server_name=config.SERVER_HOST,
        server_port=config.SERVER_PORT,
        css=custom_css,
        share=False
    )