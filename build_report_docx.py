from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
ROOT=Path(__file__).parent; OUT=ROOT/'NoduleMNIST3D_肺结节分类实验报告.docx'
doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.75); sec.bottom_margin=Inches(.75); sec.left_margin=Inches(.85); sec.right_margin=Inches(.85)
styles=doc.styles; styles['Normal'].font.name='宋体'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Normal'].font.size=Pt(10.5)
for s,size in [('Title',20),('Heading 1',15),('Heading 2',12)]:
 st=styles[s]; st.font.name='黑体'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor(0,0,0)
def shade(cell,color='D9EAF7'):
 tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),color); tcPr.append(shd)
def table(headers,rows):
 t=doc.add_table(rows=1,cols=len(headers)); t.style='Table Grid'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for c,h in zip(t.rows[0].cells,headers): c.text=str(h); shade(c); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 for row in rows:
  cells=t.add_row().cells
  for c,v in zip(cells,row): c.text=str(v); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
 return t
def para(text):
 p=doc.add_paragraph(text); p.paragraph_format.first_line_indent=Pt(21); p.paragraph_format.line_spacing=1.25; return p
def fig(path,caption,width=6.2):
 p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(ROOT/path),width=Inches(width)); q=doc.add_paragraph(caption); q.alignment=WD_ALIGN_PARAGRAPH.CENTER; q.runs[0].italic=True; q.runs[0].font.size=Pt(9)
p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('NoduleMNIST3D 肺结节分类实验报告')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('医学图像处理课程实验').bold=True
para('本实验围绕肺结节 CT 图像智能分类，比较传统图像特征、二维 TinyCNN、数据增强、类别不平衡处理、Grad-CAM 解释方法及 2.5D 输入策略。实验保留官方 train/val/test 划分，模型选择依据验证集 AUC，最终指标在官方测试集上报告。')
doc.add_heading('1 实验目的与数据',1); para('实验采用 NoduleMNIST3D 数据集。每个样本是 28×28×28 的三维灰度图像，中心切片归一化到 [0,1] 后用于二维模型。训练集、验证集和测试集分别包含 1158、165 和 310 个样本。')
table(['划分','样本数','类别 0','类别 1','类别 1 比例'],[['训练集',1158,863,295,'25.5%'],['验证集',165,123,42,'25.5%'],['测试集',310,246,64,'20.6%']]); para('训练集类别 1 明显少于类别 0，直接使用未加权损失容易使模型偏向多数类，表现为特异度高而敏感度低。')
doc.add_heading('2 任务 1 数据读取与可视化',1); para('程序检查了官方数据 MD5、三组划分、图像形状、灰度范围和类别分布。典型样本展示轴向、冠状和矢状中心切面，并绘制灰度直方图。'); fig(Path('outputs/figures/class_distribution.png'),'图 1 数据集类别分布'); fig(Path('outputs/figures/typical_views.png'),'图 2 典型体积三视图与灰度分布')
doc.add_heading('3 任务 2 传统机器学习分类',1); para('特征由 16 维灰度直方图、均值/标准差/分位数等统计量以及 14×14 下采样强度组成。分类器为带 class_weight=balanced 的 Logistic Regression，StandardScaler 只在训练集拟合。'); table(['方法','AUC','ACC','敏感度','特异度','F1'],[['传统特征 + Logistic Regression','0.871','0.829','0.781','0.841','0.654']]); para('传统模型对少数类识别较积极，敏感度高于未处理的 TinyCNN，但空间结构建模能力有限，AUC 低于类别不平衡处理后的深度模型。')
doc.add_heading('4 任务 3 TinyCNN 分类与增强',1); para('基础 TinyCNN 输入单张中心切片，包含三层二维卷积、BatchNorm、ReLU、池化和全局平均池化。优化器为 Adam，学习率 0.001，batch size 32，最大 40 个 epoch，patience=8。增强组合包括翻转、高斯噪声和随机对比度变化，验证集和测试集不增强。每个实验按验证集 AUC 保存最佳 checkpoint。'); table(['模型/增强方案','AUC','ACC','敏感度','特异度','F1'],[['TinyCNN 基线','0.898','0.858','0.313','1.000','0.476'],['TinyCNN + 翻转/噪声/对比度','0.902','0.871','0.453','0.980','0.592']]); fig(Path('outputs/figures/training_curves.png'),'图 3 TinyCNN 基线训练过程曲线'); fig(Path('outputs/figures/roc_baseline_augment.png'),'图 4 基线与增强模型 ROC 曲线'); fig(Path('outputs/figures/confusion_baseline_augment.png'),'图 5 基线与增强模型混淆矩阵'); para('增强方案提高了敏感度和 F1，同时保持较高特异度，说明输入扰动缓解了模型对多数类的偏置。')
doc.add_heading('5 任务 4 类别不平衡处理',1); para('在相同模型、最大 epoch、batch size、学习率、优化器和 patience 下比较 Weighted BCE 与随机过采样。正类权重为 863/295=2.925。两种方法都按验证集 AUC 早停，因此实际训练轮数和最终 checkpoint 轮数可能不同。'); table(['策略','AUC','ACC','敏感度','特异度','F1'],[['Weighted BCE','0.918','0.810','0.859','0.797','0.651'],['随机过采样','0.910','0.848','0.625','0.907','0.630']]); para('Weighted BCE 获得最高 AUC 和敏感度，但误报增加；随机过采样在敏感度与特异度之间更均衡。')
doc.add_heading('6 任务 5 Grad-CAM 可解释性分析',1); para('Grad-CAM 使用增强模型最后一个卷积层的特征图和目标类别梯度生成热图，分别选取 TP、TN、FP 和 FN 测试样本。热图只反映梯度敏感区域，不能单独证明模型使用了临床正确的判别依据。'); table(['案例','测试集索引','真实标签','预测概率'],[['TP',10,1,'0.723'],['TN',0,0,'0.289'],['FP',81,0,'0.784'],['FN',7,1,'0.071']])
+for name,cap in [('gradcam_TP.png','图 6 Grad-CAM TP 案例'),('gradcam_TN.png','图 7 Grad-CAM TN 案例'),('gradcam_FP.png','图 8 Grad-CAM FP 案例'),('gradcam_FN.png','图 9 Grad-CAM FN 案例')]: fig(Path('outputs/gradcam_aug')/name,cap,width=5.6)
doc.add_heading('7 选做任务 3：2.5D 分类',1); para('相邻切片方案将中心切片及前后各一张切片组成三通道输入；正交切面方案将轴向、冠状和矢状中心切面组成三通道输入。三种模型使用相同数据划分、最大训练轮数、batch size、学习率、优化器和 patience，并按验证集 AUC 选择 checkpoint。'); fig(Path('outputs/figures/25d_input_examples.png'),'图 10 三种模型输入示例'); table(['输入方式','AUC','ACC','敏感度','特异度','F1','参数量','测试推理时间'],[['单张中心切片','0.886','0.848','0.266','1.000','0.420','23585','0.026 s'],['相邻切片 2.5D','0.909','0.839','0.594','0.902','0.603','23873','0.028 s'],['正交切面 2.5D','0.886','0.858','0.609','0.923','0.639','23873','0.026 s']]); para('相邻切片输入取得最高 AUC，正交切面输入的 F1、敏感度和特异度更高，表明不同方向的形态信息具有互补性。两种 2.5D 方法只增加约 288 个参数和极少推理时间。')
doc.add_heading('8 实验结果分析与讨论',1); para('综合比较可见，Weighted BCE 的 AUC 最高，正交切面 2.5D 的 F1 最好，未处理的基线敏感度最低。模型之间的差异受到类别不平衡、中心切片信息损失、固定 0.5 阈值和样本规模影响。测试集只用于最终评价，未参与模型选择或超参数调整。')
doc.add_heading('9 实验环境与复现信息',1); para('实验在 hustvl2027 上完成，使用 NVIDIA GeForce RTX 3090、PyTorch 2.4.1 CUDA 环境。固定随机种子为 42，官方数据 MD5 校验通过，自动测试 36 项全部通过。代码、配置、权重、逐例预测、训练历史和指标 JSON 保存在 student_code/outputs。')
doc.add_heading('10 小组成员分工',1); para('以下内容由小组成员自行填写。'); table(['成员','负责内容','完成情况'],[['','',''],['','',''],['','','']])
doc.save(OUT); print(OUT)
