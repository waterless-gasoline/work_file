export type Metric = {
  label: string;
  value: string;
  detail: string;
};

export type Project = {
  slug: string;
  title: string;
  summary: string;
  scenario: string;
  techStack: string[];
  highlights: string[];
  metrics: string[];
  tags: string[];
  details: {
    background: string;
    solution: string[];
    results: string[];
  };
};

export const profile = {
  name: "彭煌城",
  role: "AI 算法 / NLP / 多模态 / 大模型方向作品集",
  intro:
    "聚焦 NLP、RAG、多模态生成、端侧部署与模型优化，擅长把模型能力落到真实业务场景，并用指标证明效果。",
  availability: ["27 届学生", "每周可出勤 5 天", "关注工程落地与结果指标"],
  contact: {
    email: "564673816@qq.com",
    phone: "13232956968",
    github: "https://github.com/waterless-gasoline/HangchengPeng",
    resume: "/resume.pdf",
  },
};

export const metrics: Metric[] = [
  {
    label: "NER F1",
    value: "92.3%",
    detail: "LoRA + 指针网络识别电商实体",
  },
  {
    label: "情感分析 F1",
    value: "93.5%",
    detail: "XLM-RoBERTa 跨语言情感分类",
  },
  {
    label: "RAG MRR@10",
    value: "0.72",
    detail: "Rerank 后检索效果显著提升",
  },
  {
    label: "端侧延迟",
    value: "<15ms",
    detail: "猫狗视觉感知系统完成边缘部署",
  },
];

export const focusAreas = [
  "NLP / 大模型应用",
  "RAG / 检索增强生成",
  "多模态生成",
  "端侧部署 / TensorRT / NCNN",
  "LoRA / 量化 / 蒸馏优化",
  "业务指标驱动建模",
];

export const projects: Project[] = [
  {
    slug: "edge-vision-system",
    title: "端侧猫狗视觉感知系统",
    summary:
      "面向智能摄像头场景，完成猫狗目标检测、实例分割与颜色识别的一体化多任务 pipeline。",
    scenario: "IPC 实时流分析，要求低延迟、低显存、稳定识别复杂环境目标。",
    techStack: [
      "PyTorch",
      "NanoDet-Plus",
      "ShuffleNetV2",
      "PicoSAM3",
      "TensorRT",
      "NCNN",
    ],
    highlights: [
      "采集 5 万+ 图像，构建遮挡、逆光、夜间 hard-case 验证集",
      "基于 SAM3 辅助标注，35 万张图片 8 小时完成，效率提升 5 倍",
      "完成检测、分割、颜色识别多任务融合与边缘部署",
    ],
    metrics: [
      "mAP@0.50:0.95 78.8%",
      "小目标 AP 46.1% → 52.3%",
      "单帧延迟 < 15ms",
      "显存 2.1GB → 0.73GB",
    ],
    tags: ["多任务学习", "边缘部署", "模型压缩"],
    details: {
      background:
        "项目面向 IPC 智能摄像头，希望在同一套轻量模型中同时完成检测、分割和颜色识别，降低推理资源开销并提升上线可用性。",
      solution: [
        "采用 NanoDet-Plus + ShuffleNetV2 作为轻量检测主干，并针对小目标引入特征金字塔优化。",
        "利用 PicoSAM3 蒸馏实例分割能力，同时保持边缘设备可运行的计算量。",
        "通过 INT8/FP16 量化、TensorRT 加速和 NCNN 部署打通端到端落地链路。",
      ],
      results: [
        "检测精度达到 mAP@0.50:0.95 = 78.8%，AP@0.50 = 94.3%。",
        "单任务模型相比，多任务方案推理延迟降低 60%，显存占用减少 50% 以上。",
        "在主流边缘模组上达到单帧 < 15ms，满足实时流分析需求。",
      ],
    },
  },
  {
    slug: "rag-ecommerce-qa",
    title: "RAG 电商智能问答系统",
    summary:
      "构建向量检索 + BM25 + Rerank 的电商问答系统，解决专有名词识别与语义泛化问题。",
    scenario: "面向商品咨询、订单问题、售后问答等客服场景。",
    techStack: [
      "ChatGLM3",
      "Milvus",
      "Elasticsearch",
      "BM25",
      "BGE-Reranker",
      "LangGraph",
    ],
    highlights: [
      "构建 10 万+ FAQ 知识库与父子文档索引",
      "结合向量检索、关键词召回与 Rerank 精排优化结果质量",
      "使用 LoRA 微调 ChatGLM3，贴合电商客服语气与业务约束",
    ],
    metrics: [
      "召回率提升 23%",
      "MRR@10 0.58 → 0.72",
      "回答准确率 72% → 91%",
      "客服工作量减少 30%",
    ],
    tags: ["RAG", "检索优化", "大模型应用"],
    details: {
      background:
        "项目目标是解决纯关键词检索在电商问答中的召回不足和泛化差问题，同时让生成回答更符合客服场景要求。",
      solution: [
        "基于 Milvus 和 Elasticsearch 构建混合检索体系，兼顾语义匹配与专有词命中。",
        "对 Top-50 粗排结果用 BGE-Reranker 精排，并用 LangGraph 路由不同业务意图。",
        "对 ChatGLM3 做 LoRA 指令微调，提升客服口吻一致性和业务回答准确率。",
      ],
      results: [
        "Rerank 后 MRR@10 从 0.58 提升到 0.72。",
        "离线评测集回答准确率从 72% 提升到 91%。",
        "系统上线后显著减轻客服重复问答压力。",
      ],
    },
  },
  {
    slug: "lora-ner-system",
    title: "LoRA 电商命名实体识别系统",
    summary:
      "针对品牌、商品、属性、价格等实体识别需求，构建轻量高效的电商 NER 系统。",
    scenario: "面向商品描述、用户评论、知识图谱构建等电商文本理解任务。",
    techStack: ["PyTorch", "BERT", "LoRA", "Jieba", "BIO 标注", "指针网络"],
    highlights: [
      "处理 15 万条电商文本，构建 12 万标注实体数据",
      "在 Q/V 矩阵注入低秩适配模块，降低训练成本",
      "用指针网络替代 CRF，提升嵌套实体与非连续实体识别能力",
    ],
    metrics: [
      "F1 92.3%",
      "准确率 93.5%",
      "训练参数量减少 80%",
      "推理速度提升 40%",
    ],
    tags: ["NER", "LoRA 微调", "信息抽取"],
    details: {
      background:
        "项目针对电商 SKU 文本中品牌、型号、规格强耦合的特点，重点解决嵌套实体和复杂边界识别问题。",
      solution: [
        "以 BERT 为底座，通过 LoRA 做参数高效微调，减少训练资源消耗。",
        "通过消融实验确定 r=8、α=16 的较优组合。",
        "将传统 CRF 解码替换为指针网络，提升复杂实体结构下的识别效果。",
      ],
      results: [
        "测试集 F1 达到 92.3%，准确率 93.5%，召回率 91.2%。",
        "相较全参数微调，训练参数量减少 80%，训练时间缩短 50%。",
        "识别结果成功用于电商知识图谱构建，并带动下游推荐 CTR 提升 5%。",
      ],
    },
  },
  {
    slug: "multilingual-sentiment",
    title: "跨语言用户评论情感分析系统",
    summary:
      "使用 XLM-RoBERTa 做英泰等多语言情感分类，提升跨境电商评论理解能力。",
    scenario: "面向东南亚市场的评论分析、纠纷处理和用户满意度监控。",
    techStack: ["PyTorch", "XLM-RoBERTa", "BERT", "FastText", "Word2Vec"],
    highlights: [
      "构建英泰对齐评论数据，缓解小语种标注不足问题",
      "对比多种深度学习模型并完成超参数优化",
      "提升自动化纠纷处理与商家运营反馈效率",
    ],
    metrics: [
      "F1 93.5%",
      "准确率 94.2%",
      "跨语言 F1 提升 12%",
      "退货率降低 8%",
    ],
    tags: ["跨语言 NLP", "情感分析", "电商场景"],
    details: {
      background:
        "项目聚焦东南亚多语言电商评论，希望在小语种标注数据有限的前提下，稳定识别用户情绪和纠纷风险。",
      solution: [
        "以 XLM-RoBERTa 作为跨语言 backbone，结合迁移学习提升低资源语言表现。",
        "通过数据增强、过采样与超参数优化缓解类别不平衡问题。",
        "对比 LSTM、GRU、FastText、BERT 等模型，最终选择效果最优方案。",
      ],
      results: [
        "测试集准确率 94.2%，F1 93.5%。",
        "相较普通跨语言基线，F1 提升 12%。",
        "帮助自动化纠纷处理效率提升 40%，并降低退货率 8%。",
      ],
    },
  },
  {
    slug: "multimodal-captioning",
    title: "多模态商品图像描述生成系统",
    summary:
      "构建图生文模型自动生成商品描述，提升跨境电商上架与运营效率。",
    scenario: "面向商品详情页文案生成与多语言电商内容生产。",
    techStack: [
      "TensorFlow",
      "Keras",
      "Inception V3",
      "Attention",
      "Top-p Sampling",
      "对比搜索",
    ],
    highlights: [
      "构建 5 万+ 图文对数据集，完成图像与文本联合建模",
      "引入目标检测属性注入，减少描述幻觉问题",
      "结合 Top-p Sampling 与对比搜索提升生成多样性与可读性",
    ],
    metrics: [
      "BLEU-4 0.45",
      "CIDEr 2.8",
      "CTR 提升 20%",
      "单 SKU 耗时缩短至 3 秒",
    ],
    tags: ["多模态", "AIGC", "图生文"],
    details: {
      background:
        "项目目标是替代人工商品文案撰写，让电商团队可以批量生成更贴合图片内容的商品描述。",
      solution: [
        "用 Inception V3 编码图像特征，并通过注意力机制驱动文本生成。",
        "将颜色、材质、领型等目标检测属性注入解码器上下文，减少描述与实物不符的问题。",
        "使用 Teacher Forcing、Top-p Sampling 与对比搜索平衡收敛速度和生成质量。",
      ],
      results: [
        "BLEU-4 达到 0.45，较常见基线提升明显。",
        "AIGC 描述带动 TikTok 泰国区测试 CTR 提升 20%。",
        "单 SKU 文案生成从 20 分钟缩短到 3 秒，大幅降低运营成本。",
      ],
    },
  },
];
