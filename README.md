大模型应用开发辅助工具集

LLM Application Development Toolkit

项目简介 | Project Introduction

面向 AI 大模型应用开发者的一站式效率工具，集成 Prompt 智能优化、数据标准化清洗、机器学习数据集拆分、可视化交互操作等核心能力，覆盖大模型落地全过程的数据预处理与工程化基础工作，无需命令行即可完成全流程操作，大幅降低 AI 开发门槛。

An all-in-one efficiency toolkit for AI large language model developers. It integrates core capabilities including intelligent Prompt optimization, standardized data cleaning, machine learning dataset splitting, and visual interactive operations. Covering the entire process of data preprocessing and engineering basics for LLM deployment, it supports full-process operations without command line, greatly reducing the barrier to AI development.

技术栈 | Tech Stack
开发语言：Python
可视化框架：Gradio
大模型接口：阿里云通义千问 API
数据处理：Pandas、NumPy
存储与工程：SQLite、日志系统、全局异常捕获

核心功能 | Core Features
Prompt 智能优化自动结构化优化提示词，输出规范、精准、可直接复用的专业 Prompt。Intelligent Prompt optimization automatically structures and improves prompts, generating standardized, accurate, ready-to-use professional prompts.
数据标准化处理支持 CSV 文件一键清洗，自动处理缺失值、重复值、特殊字符，兼容 GBK/UTF-8 多编码。One-click CSV data cleaning, automatically handling missing values, duplicates, special characters, and supporting GBK/UTF-8 encodings.
数据集智能拆分支持训练集 / 验证集 / 测试集按比例自动拆分，固定随机种子保证实验可复现，符合大模型微调标准。Intelligent train/validation/test dataset splitting with fixed random seed to ensure reproducibility, fully compliant with LLM fine-tuning standards.
可视化交互界面基于 Gradio 搭建深色主题可视化界面，全流程图形化操作，简洁易用。Dark-themed visual interface built with Gradio, providing full-process graphical operation with simple and intuitive experience.
工程化规范保障全局异常捕获、日志记录、API 密钥安全管理，代码分层清晰，健壮性强。Global exception catching, logging, secure API key management, clean code architecture, and high robustness.

快速开始 | Quick Start
安装依赖：pip install gradio pandas numpy dashscope tenacity
运行：python main.py
访问地址:浏览器打开：http://localhost:7860

使用说明 | Usage Guide
在「系统设置」配置阿里云通义千问 API 密钥
使用「Prompt 智能优化」生成高质量提示词
使用「数据预处理」完成数据清洗与数据集拆分
所有结果自动保存到 ./output 目录

Configure your Alibaba Cloud Tongyi Qianwen API Key in Settings
Use Prompt Optimization to generate high-quality prompts
Use Data Preprocessing for data cleaning and dataset splitting
All results are auto-saved to the ./output directory

项目亮点 | Project Highlights
开箱即用，无需命令行基础
完整覆盖大模型落地前数据与 Prompt 工程流程
界面美观、交互流畅、工程化规范完善
自动处理编码、异常、日志等工程细节
轻量化部署，本地一键启动

Lightweight, out-of-the-box, full-process LLM preprocessing toolkit with clean UI and professional engineering standards.

备注 | Note
本项目为个人学习与工程实践作品，仅用于学习、实践展示，请勿用于商业用途。
This project is for personal study, engineering practice, resume demonstration only. Not for commercial use.
