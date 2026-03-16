## FAKESHIELD: EXPLAINABLE IMAGE FORGERY DE-

## TECTION AND LOCALIZATION VIA MULTI-MODAL

## LARGELANGUAGE MODELS

```
Zhipei Xu^1 ,^2 , Xuanyu Zhang^1 , Runyi Li^1 , Zecheng Tang^1 , Qing Huang^4 , Jian Zhang^1 ,^2 ,^3 †
```
(^1) School of Electronic and Computer Engineering, Peking University
(^2) Peking University Shenzhen Graduate School-Rabbitpre AIGC Joint Research Laboratory
(^3) Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology,
Shenzhen Graduate School, Peking University
(^4) School of Future Technology, South China University of Technology

## ABSTRACT

```
The rapid development of generative AI is a double-edged sword, which not only
facilitates content creation but also makes image manipulation easier and more dif-
ficult to detect. Although current image forgery detection and localization (IFDL)
methods are generally effective, they tend to face two challenges:1)black-box
nature with unknown detection principle,2)limited generalization across diverse
tampering methods (e.g., Photoshop, DeepFake, AIGC-Editing). To address these
issues, we propose the explainable IFDL task and design FakeShield, a multi-
modal framework capable of evaluating image authenticity, generating tampered
region masks, and providing a judgment basis based on pixel-level and image-
level tampering clues. Additionally, we leverage GPT-4o to enhance existing
IFDL datasets, creating the Multi-Modal Tamper Description dataSet (MMTD-
Set) for training FakeShield’s tampering analysis capabilities. Meanwhile, we
incorporate a Domain Tag-guided Explainable Forgery Detection Module (DTE-
FDM) and a Multi-modal Forgery Localization Module (MFLM) to address var-
ious types of tamper detection interpretation and achieve forgery localization
guided by detailed textual descriptions. Extensive experiments demonstrate that
FakeShield effectively detects and localizes various tampering techniques, offer-
ing an explainable and superior solution compared to previous IFDL methods.
The code is available athttps://github.com/zhipeixu/FakeShield.
```
## 1 INTRODUCTION

```
With the rapid development of AIGC, powerful image editing models have provided a breeding
ground for convenient image tampering, blurring the boundaries between true and forgery. People
can use AIGC image editing methods (Rombach et al., 2022; Zhang et al., 2023; Suvorov et al.,
2022; Mou et al., 2023) to edit images without leaving a trace. Although it has facilitated the work
of photographers and illustrators, AIGC editing methods have also led to an increase in malicious
tampering and illegal theft. The authenticity of images in social media is difficult to guarantee,
which will lead to problems such as rumor storms, economic losses, and legal concerns. Therefore,
it is important and urgent to identify the authenticity of images. In this context, the image forgery
detection and localization (IFDL) task aims to identify whether an image has been tampered with
and locate the specific manipulation areas. It can be widely applied in the real world, such as filtering
false content on social media, preventing the spread of fake news, and court evidence collection.
```
```
State-of-the-art IFDL methods have utilized well-designed network structures, elaborate network
constraints, and efficient pre-training strategies to achieve remarkable performance (Yu et al., 2024b;
Ma et al., 2023; Dong et al., 2022). However, previous IFDL methods face two key problems, lim-
```
```
†Corresponding author: Jian Zhang. This work was supported in part by Guangdong Provincial Key
Laboratory of Ultra High Definition Immersive Media Technology (No. 2024B1212010006) and Shenzhen
General Research Project (No. JCYJ20241202125904007).
```
# arXiv:2410.02761v4 [cs.CV] 12 Apr 2025


```
Detector
```
```
Locator
```
```
Confidence
```
```
The picture been tampered withhas ,
especially in the center of the image.
there is a penguin
standing on the tatami floor
```
```
The perspective and scale
relationship penguin and the room appears between the
unnatural being front , with the penguin - lit while the room
is illuminated by diffuse light.
These discrepancies suggest potential tampering with the
image.
```
```
Can you identify
manipulated areas
in the photograph?
Tampered area mask
```
```
Detect result Principle explanation
(a) Conventional IFDL Framework (b) Our Explainable IFDL Framework: FakeShield
```
```
Tampered area mask
```
```
Principle explanation
Multimodal interaction
```
```
Input image
```
```
Input image Instruction
```
```
FakeShield
```
```
True/false detection
Tampered area location
```
```
Principle explanation
Multimodal interaction
```
```
True/false detection
Tampered area location
```
Figure 1: Illustration of the conventional IFDL and explainable IFDL framework. Conventional
methods offer only detection results and tampered masks. We extend this into a multi-modal frame-
work, enabling detailed explanations and conversational interactions for a deeper analysis.

iting their practicality and generalizability. First, as shown in Figure 1(a), most existing IFDL
methods are black-box models, only providing the authenticity probability of the image, while the
principle of detection is unknown to users. Since the existing IFDL methods cannot guarantee
satisfactory accuracy, manual subsequent judgment is still required. Given that the information
provided by the IFDL methods is insufficient, it is difficult to support the human assessment and
users still need to re-analyze the suspect image by themselves. Second, in real-world scenarios,
tampering types are highly diverse, including Photoshop (copy-and-move, splicing, and removal),
AIGC-Editing, DeepFake, and so on. Existing IFDL methods (Yu et al., 2024b; Ma et al., 2023) are
typically limited to handling only one of these techniques, lacking the ability to achieve compre-
hensive generalization. This forces users to identify different tampering types in advance and apply
specific detection methods accordingly, significantly reducing these models’ practical utility.

Benefiting from the rapid advancements in Transformer architectures, Large Language Models
(LLMs) have attracted significant attention. Furthermore, (Liu et al., 2024) introduced a Multi-
modal Large Language Model (M-LLM) that aligns visual and textual features, thereby endowing
LLMs with enhanced visual comprehension abilities. Given that LLMs are pre-trained on an ex-
tensive and diverse corpus of world knowledge, they hold significant potential for a wide range
of applications, such as machine translation (Devlin, 2018), code completion, and visual under-
standing (Liu et al., 2024). Consequently, we explored the feasibility of employing M-LLMs for
explainable Image Forgery Detection and Localization (e-IFDL). This approach allows for a more
comprehensive explanation of the rationale behind tampering detection and provides a more precise
identification of both the authenticity of images and the suspected manipulation regions.

To address the two issues of the existing IFDL methods, we propose the explainable-IFDL (e-IFDL)
task and a multi-modal explainable tamper detection framework called FakeShield. As illustrated
in Figure1(b), the e-IFDL task requires the model to evaluate the authenticity of any given image,
generate a mask for the suspected tampered regions, and provide a rationale based on some pixel-
level artifact details (e.g., object edges, resolution consistency) and image-level semantic-related
errors (e.g., physical laws, perspective relationships). Leveraging the capabilities of GPT-4o (Ope-
nAI, 2023), we can generate a comprehensive triplet consisting of a tampered image, a modified
area mask, and a detailed description of the edited region through a meticulously crafted prompt.
Then, we develop theMulti-ModalTamperDescription dataSet(MMTD-Set) by building upon ex-
isting IFDL datasets. Utilizing the MMTD-Set, we fine-tune M-LLM (Liu et al., 2024) and visual
segmentation models (Kirillov et al., 2023; Lai et al., 2024), equipping them with the capability
to provide complete analysis for judgment, detecting tampering, and generate accurate tamper area
masks. This process ultimately forms a comprehensive forensic pipeline for analysis, detection, and
localization. Our contributions are summarized as follows:

❑(1) We present the first attempt to propose a multi-modal large image forgery detection and local-
ization model, dubbedFakeShield. It can not only decouple the detection and localization process
but also provide a reasonable judgment basis, which alleviates the black-box property and unex-
plainable issue of existing IFDL methods.

❑(2) We use GPT-4o to enrich the existing IFDL dataset with textual information, constructing the
MMTD-Set. By guiding it to focus on distinct features for various types of tampered data, GPT-4o
can analyze the characteristics of tampered images and construct ”image-mask-description” triplets.


❑(3) We develop aDomainTag-guidedExplainableForgeryDetectionModule (DTE-FDM) to
spot different types of fake images in a united model and effectively alleviate the data domain con-
flict. Meanwhile, anMulti-modalForgeryLocalizationModule (MFLM) is adopted to align visual-
language features, thus pinpointing tampered areas.

❑(4) Extensive experiments demonstrate that our method can accurately analyze tampering clues,
and surpass most previous IFDL methods in the detection and localization of many tampering types
like copy-move, splicing, removal, DeepFake, and AIGC-based editing.

## 2 RELATED WORKS

#### 2.1 IMAGEFORGERYDETECTION ANDLOCALIZATION

Prevailing IFDL methods mainly target at the localization of specific manipulation types (Salloum
et al., 2018; Islam et al., 2020; Li & Zhou, 2018; Zhu et al., 2018; Li & Huang, 2019). In con-
trast, universal tamper localization methods (Li et al., 2018; Kwon et al., 2021; Chen et al., 2021;
Ying et al., 2023; 2021; Hu et al., 2023; Ying et al., 2022; Li et al., 2024; Yu et al., 2024a; Zhang
et al., 2024b) aim to detect artifacts and irregularities across a broader spectrum of tampered images.
For instance, MVSS-Net (Dong et al., 2022) utilized multi-scale supervision and multi-view feature
learning to simultaneously capture image noise and boundary artifacts. OSN (Wu et al., 2022)
employed a robust training strategy to overcome the difficulties associated with lossy image pro-
cessing. HiFi-Net (Guo et al., 2023) adopted a combination of multi-branch feature extraction and
localization modules to effectively address alterations in images synthesized and edited by CNNs.
IML-ViT (Ma et al., 2023) integrated Swin-ViT into the IFDL task, employing an FPN architecture
and edge loss constraints to enhance its performance. DiffForensics (Yu et al., 2024b) adopted a
training approach akin to diffusion models, strengthening the model’s capacity to capture fine image
details. Additionally, some researchers (Zhang et al., 2024a;c; Asnani et al., 2023) have pursued
proactive tamper detection and localization by embedding copyright and location watermarks into
images/audio/videos preemptively. However, despite their acceptable performances, these IFDL
methods cannot explain the underlying principles and rationale behind their detection and local-
ization judgments, offering no interaction. Moreover, they suffer from limited generalization and
accuracy, exhibiting significant performance disparities across different testing data domains.

#### 2.2 LARGELANGUAGEMODEL

Large language models (Dubey et al., 2024; OpenAI, 2023) have garnered global attention in re-
cent years for their exceptional instruction-following and text-generation abilities. Based on the
Transformer architecture, LLMs are pre-trained on massive datasets, allowing them to accumulate
broad world knowledge that enhances their ability to generalize across a wide range of downstream
tasks. Subsequently, some researchers (Li et al., 2022) expanded LLMs’ powerful understanding
and world knowledge to the visual domain by incorporating image encoders and projection layers,
which enable images encoded into tokens that align with the text. Some recent works (Chen et al.,
2023a; Wang et al., 2023; Chen et al., 2023b) equipped M-LLMs with enhanced visual understand-
ing capabilities by expanding the visual instruction datasets and increasing the model size during
fine-tuning. Currently, M-LLMs demonstrate impressive performance across various downstream
tasks. LISA (Lai et al., 2024) integrated SAM (Kirillov et al., 2023) with M-LLM to implement rea-
soning segmentation, enabling the generation of masks from text descriptions. GLaMM (Rasheed
et al., 2024) further enhanced this by using a more advanced region image encoder to improve text-
to-mask grounding. Additionally, some studies (Yang & Zhou, 2024; Zhang et al., 2024d) have
explored the application of M-LLMs in DeepFake detection. For instance, (Zhang et al., 2024d)
introduced the DD-VQA dataset, combining a manual inference process for rating real and fake
faces that can be distinguished using common sense. Targeted at Deepfake detection, (Huang et al.,
2024) used GPT-4o to create image-analysis pairs, and introduced a multi-answer intelligent deci-
sion system into MLLM, achieving good effect. However, it cannot be generalized to other types of
tampering such as Photoshop and AIGC-Editing, and cannot accurately locate the tampered areas.
Besides, using M-LLMs to realize universal tamper localization and detection remains unexplored.


```
USER: this image ... Has
been forged with? Why?
GPT response
```
```
Tamper type
```
```
CASIAv2 FFHQ
```
```
Fantasitic Reality (FR) FaceAPP
```
```
Source Datasets
```
```
Au-Scene PhotoShop Tp-Scene DeepFake Tp-Face
```
```
Lighting
```
```
Image
```
```
Reference
```
```
Query: You are an AI visual assistant that can help humans analyze some tampered images.
Now your task is to use the tampered picture A When describing the image, here are some angles you need to pay special attention to...and binary mask of the tampered area B to...
```
```
Tamper Type Specific
Prompts
PhotoShop TamperDetect: The picture has been tampered with. ：
Locate: Judgment basis: The tampered area is located...
```
```
GPT-4o
```
```
No ref No ref
```
```
Common Sense Resolution Edge
Copy Paste Physical Laws Perspective Text
AIGCDetect: -Editing Tamper The picture has been tampered with. ：
Locate: Judgment basis: The tampered area is located...
```
```
Au-Face
```
```
USER: this image ... Has
been forged with? Why?
```
```
Au-Scene: Real scene images GPT response
from FR,CASIAv2 and COCO MMTD-Set
```
```
COCO
```
```
AIGC-Editing
```
```
AIGCTp--SceneEditing
```
```
DeepFake TamperDetect: The picture has been tampered with. ：
Judgment basis:
Symmetry Expressions Hairstyle Text
Makeup Accessories Common Sense Blur
```
```
Type
```
```
PhotoShop Tp-Scene: Tampered
scene images from FR, CASIAv
```
```
AIGC-Editing Tp-Scene: Scene images
tampered by AIGC-Editing methods
Au-Face: Real face images
from FFHQ
```
```
DeepFake Tp-Face: Tampered face
images from FaceAPP
```
```
Expert
Proofreading
```
```
Texture EdgeCommon Sense Lighting
Detail LossBlur Anatomy Error Repetition
```
Figure 2: Illustration of the construction process of our MMTD-Set. We sample the tampered image-
mask pairs from PS, DeepFake, and AIGC benchmarks, and then use domain tags to guide GPT-4o
in constructing the judgment basis and focusing on both pixel-level details and image-level content.

## 3 METHODOLOGY

#### 3.1 CONSTRUCTION OF THE PROPOSEDMMTD-SET

Motivation:Most existing IFDL datasets consist of a single visual modality, lacking training visual-
language samples adapted to M-LLMs. The challenge of constructing our MMTD-Set lies in ac-
curately translating the visual tampering information from the existing IFDL image datasets into
precise textual descriptions. To address this challenge, our core contributions focus on two aspects:
(1)We leverage GPT-4o to generate text description and provide both the tampered image and its
corresponding mask to GPT-4o, enabling it to accurately identify the tampered location.(2)For each
tamper type, we design specific prompts to their unique characteristics, guiding GPT-4o to focus on
different tampering artifacts and providing more detailed visual cues.

Data collection:Based on (Ma et al., 2023; Nirkin et al., 2021), we categorize common tamper-
ing into three types: PhotoShop (copy-move, splicing, removal), DeepFake (FaceAPP (FaceApp
Limited, 2017)), and AIGC-Editing (SD-inpainting (Lugmayr et al., 2022)). As shown in Figure 2,
we gathered three types of tampered images along with their corresponding authentic images from
public datasets (Dong et al., 2013; Dang et al., 2020) and self-constructed data.

GPT assisted description generation: Given that manual analysis of tampered images is time-
consuming, inspired by (Liu et al., 2024; Chen et al., 2023a; Huang et al., 2024), we used GPT-4o to
automate the analysis of tampered images. As depicted in Figure 2, the output analysis is required
to follow the format of detected results, localization descriptions, and judgment basis.

For tampered images, we input the edited image, its corresponding forgery mask, and our carefully
constructed tamper type specific prompts into the powerful GPT-4o to more accurately describe the
tampered regions.For authentic images, GPT-4o is provided with only the real image and a set of
prompts, guiding it to confirm its authenticity.The full-text prompts are detailed in the Appendix A.6.
To more clearly and specifically describe and analyze the tampering of images, GPT-4o describes the
image from two key aspects: the location and content of the tampered areas, and any visible artifacts
or semantic errors caused by the tampering:(1)For the tampering location, GPT-4o is required to
describe it in both absolute positions (e.g., top, bottom, upper left corner, lower right corner) and
relative positions (e.g., above the crowd, on the table, under the tree). When analyzing the tampered
content, it is tasked with providing detailed information about the types, quantities, actions, and
attributes of the objects within the tampered region.(2)For the visible artifacts and semantic errors,
since different tampering methods produce distinct types of artifacts, we craft specific prompts to
guide the analysis. It can broadly be categorized into pixel-level artifact details and image-level
semantic-related errors. For PhotoShop (PS) tampering, operations like copy-move and splicing
often introduce pixel-level issues such as edge artifacts, abnormal resolution, and inconsistencies
in lighting. Additionally, semantic-level errors, including violations of physical laws or common
sense, are frequently observed. In AIGC-Editing (AIGC), for instance, it often fails to generate text


```
Image
Encoder
```
```
Large Language Model
```
```
“Can you identify manipulated
areas in the photograph?”
```
```
“The picture has been tampered with,
especially in the center of the image......”
```
```
Domain Tag
Generator
```
```
FC Layer
```
```
Tamper
Comprehension Module
```
```
Full Parameter Training
```
```
[IMG 1 ], [IMG 2 ], ..., [IMG𝑘]
```
```
LoRA Finetune Frozen
```
```
Segment Anything Model
```
```
“This is a suspected
{PS/DeepFake/AIGC}
tampered picture.”
```
Figure 3: The pipeline of FakeShield. Given an imageIorifor detection, it is first processed by
the Domain Tag GeneratorGdtto obtain a data domain tagTtag. The tagTtag, along with the
text instructionTinsand image tokensTimg, are simultaneously input into the fine-tuned LLM,
generating tamper detection result and explanationOdet. Subsequently,OdetandTimgare input
into the Tamper Comprehension ModuleCt, and the last-layer embedding for the<SEG>token
h<SEG>serves as a prompt for SAM, guiding it to generate the tamper area maskMloc.

accurately, resulting in disordered symbols or characters appearing in the tampered area. For the
DeepFake (DF), tampering with facial features frequently results in localized blurring.

#### 3.2 OVERALLFRAMEWORK OFFAKESHIELD

Our goals involve two issues:1):Utilizing the textual understanding ability and world knowledge
of the M-LLM to analyze and judge the authenticity of images;2):Adopting the analysis and in-
terpretation of tampered images to assist the segmentation model in pinpointing the tampered areas.
To solve these two tasks, an intuitive approach is to fine-tune a large multimodal model to simulta-
neously output analysis and tampered masks. However, we find that joint training of multiple tasks
will increase the difficulty of network optimization and interfere with each other. Considering that
detection and interpretation focus more on language understanding and organization, while localiza-
tion requires more accumulation of visual prior information, the proposed FakeShield contains two
key decoupled parts, namely DTE-FDM and MFLM, as illustrated in Fig. 3. Specifically, an original
suspected imageIoriand an instruction textTins(e.g. “Can you identify manipulated areas in the
photograph?”) are fed to the proposed DTE-FDM to predict the detection result and judgment basis
Odet. In this process, we use a learnable generator to produce a domain tagTtag, thus avoiding the
tampered data domain conflict. Furthermore, we input the interpretationOdetand the imageIorito
the MFLM to accurately extract the tampered maskMloc. To promote cross-modal interaction for
tamper localization, we introduce a tamper comprehension module to align the visual and textual
features and enhance the ability of the vision foundation model to understand long descriptions.

#### 3.3 DOMAINTAG-GUIDEDEXPLAINABLEFORGERYDETECTIONMODULE

Motivation:In real-life scenarios, images can be tampered with and attacked through various meth-
ods, including copy-move, splicing, removal, DeepFake, and AIGC-based methods. However, these
tampered images have different distribution characteristics, and domain differences, making it dif-
ficult to apply a single IFDL method to all forgery data. For example, DeepFake focuses on face
modification, often causing partial blurring and unnatural features in the lips, teeth, and eyes. In
contrast, tools like PhotoShop (splicing, copy-move, removal) tend to leave noticeable artifacts at
the edges of the tampered areas. In the case of AIGC-Editing, blurring within the tampered region
often alters or obscures texture details. To mitigate these significant domain discrepancies, inspired


by (Sanh et al., 2022), we introduce theDomain Tag Generator(DTG), which utilizes a specialized
domain tag to prompt the model to distinguish between various data domains.

First, the original imageIoriis input into a classifierGdtto obtain the domain tagTtag. Specifically,
we classify all common tampering types into three categories: Photoshop-based editing, DeepFake,
and AIGC-based tampering, and use the template“This is a suspected{data domain}-tampered
picture.”as the identifier. Simultaneously, consistent with (Liu et al., 2024),Ioriis passed through
the image encoderFencand linear projection layerFprojto generate the image tokens [IMG]Timg.
Next,TtagandTimgare concatenated with the instructionTinsand then fed into theLLM. To
be noted,Tinsis a prompt that instructs the model to detect tampering and describe the location of
the manipulation, for example:“Can you identify manipulated areas in the photograph?”. After
several autoregressive predictions, the outputOdetcomprises three components: detection results, a
description of the location of tampered area, and the interpretive basis for the detection.

```
Ttag=Gdt(Iori), Timg=Fproj(Fenc(Iori)) (1)
Odet= LLM(Tins,Ttag|Timg). (2)
```
Given the large size of LLMs and limited computational resources, full parameter training is im-
practical. Thus, we freeze the LLM and leverage LoRA fine-tuning technology (Hu et al., 2022) to
preserve semantic integrity while enabling efficient image forgery detection.

#### 3.4 MULTI-MODALFORGERYLOCALIZATIONMODULE

Motivation: AlthoughOdetprovides a textual description of the tampered area, it lacks preci-
sion and intuitive clarity. To address this issue, we aim to transformOdetinto an accurate binary
mask, providing a clearer and more accurate representation of the tampered region. Existing prompt-
guided segmentation algorithms (Kirillov et al., 2023; Lai et al., 2024) struggle to capture the seman-
tics of long texts and hard to accurately delineate modified regions based on detailed descriptions.
Inspired by (Lai et al., 2024), we propose aTamper Comprehension Module (TCM), which is
an LLM serving as an encoder aligns long-text features with visual modalities, enhancing SAM’s
precision in locating the forgery areas. To generate the prompt fed into SAM, following (Lai et al.,
2024), we introduce a specialized token<SEG>.

As shown in Fig.3, the tokenized imageTimgand the tampered descriptionOdetare fed into the
TCMCt. Then, we extract the last-layer embedding of TCM and transform it intoh<SEG>via an MLP
projection layer. Simultaneously, the original imageIoriis processed through the SAM encoderSenc
and decoderSdec, whereh<SEG>serve as a prompt forSdecguiding the mask generationMloc.

```
Emid=Senc(Iori), h<SEG>= Extract(Ct(Timg,Odet))
Mloc=Sdec(Emid|h<SEG>),
```
#### (3)

whereEmidrepresents the intermediate features of SAM, andExtract(·)denotes the operation of
extracting the last-layer embedding corresponding to the<SEG>token. Similar to DTE-FDM, we
also apply LoRA fine-tuning to MFLM for greater efficiency. With the integration of TCM, SAM
will achieve more precise localization of the forgery areas.

#### 3.5 TRAININGOBJECTIVES

The two submodules of our FakeShield are trained end-to-end separately. For DTE-FDM, the
domain tag generator utilizes cross-entropy lossℓceas its training objective, enabling it to distinguish
between different data domains. Following the approach of LLaVA, our LLM’s training objective is
the cross-entropy lossℓce. The training target of DTE-FDMℓdetcan be formulated as:

```
ℓdet=ℓce(Oˆdet,Odet) +λ·ℓce(Tˆtag,Ttag), (4)
```
whereλdenotes the weight balancing different loss components,OdetandTtagrepresent the pre-

dictions of LLM and DTG, whileOˆdetandTˆtagrepresent their corresponding ground truth.For
MFLM, we applyℓceto constrain TCM to produce high-quality promptytxtwith<SEG>token.
Meanwhile, we use a linear combination of binary cross-entropy lossℓbceand dice lossℓdiceto en-
courage the output of MFLMMlocto be close to the GT maskMˆloc. Given the ground-truth prompt

yˆtxt(e.g., “It is<SEG>”) and maskMˆloc, our training losses for MFLMℓloccan be formulated as:

```
ℓloc=ℓce(yˆtxt,ytxt) +α·ℓbce(Mˆloc,Mloc) +β·ℓdice(Mˆloc,Mloc), (5)
```

Table 1: Detection performance comparison between our FakeShield and other competitive methods.
Our method can achieve the best detection accuracy in PhotoShop, DeepFake, and AIGC-Editing
tampered datasets. The best score is highlighted inboldand the second-best score is underlined.

```
Method
```
```
PhotoShop DeepFake AIGC-Editing
CASIA1+ IMD2020 Columbia Coverage DSO
ACC F1 ACC F1 ACC F1 ACC F1 ACC F1 ACC F1 ACC F
SPAN 0.60 0.44 0.70 0.81 0.87 0.93 0.24 0.39 0.35 0.52 0.78 0.78 0.47 0.
ManTraNet 0.52 0.68 0.75 0.85 0.95 0.97 0.95 0.97 0.90 0.95 0.50 0.67 0.50 0.
HiFi-Net 0.46 0.44 0.62 0.75 0.68 0.81 0.34 0.51 0.96 0.98 0.56 0.61 0.49 0.
PSCC-Net 0.90 0.89 0.67 0.78 0.78 0.87 0.84 0.91 0.66 0.80 0.48 0.58 0.49 0.
CAT-Net 0.88 0.87 0.68 0.79 0.89 0.94 0.23 0.37 0.86 0.92 0.85 0.84 0.82 0.
MVSS-Net 0.62 0.76 0.75 0.85 0.94 0.97 0.65 0.79 0.96 0.98 0.84 0.91 0.44 0.
FakeShield 0.95 0.95 0.83 0.90 0.98 0.99 0.97 0.98 0.97 0.98 0.93 0.93 0.98 0.
```
whereαandβare weighting factors used to balance the respective losses.ℓce,ℓbce, andℓdicerefer
to cross-entropy loss, binary cross-entropy loss, and dice loss (Sudre et al., 2017) respectively.

## 4 EXPERIMENT

#### 4.1 EXPERIMENTALSETUP

Dataset: We employ the dataset construction method outlined in Section 3.1 to build the train-
ing and test sets of the MMTD-Set. For the training set, we utilize PhotoShop tampering (e.g.,
CASIAv2 (Dong et al., 2013), Fantastic Reality (Kniaz et al., 2019)), DeepFake tampering (e.g.,
FFHQ, FaceApp (Dang et al., 2020)), and some self-constructed AIGC-Editing tampered data as
the source dataset. For the testing set, we select several challenging public benchmark datasets
including PhotoShop tampering (CASIA1+ (Dong et al., 2013), Columbia (Ng et al., 2009),
IMD2020 (Novozamsky et al., 2020), Coverage (Wen et al., 2016), DSO (De Carvalho et al., 2013),
Korus (Korus & Huang, 2016)), DeepFake tampering (e.g., FFHQ, FaceApp (Dang et al., 2020),
Seq-DeepFake (Shao et al., 2022)), and some self-generated AIGC-Editing data.

State-of-the-Art Methods:To ensure a fair comparison, we select competitive methods that provide
either open-source code or pre-trained models. To evaluate theIFDL performanceof FakeShield,
we compare it against SPAN (Hu et al., 2020), MantraNet (Wu et al., 2019), OSN (Wu et al., 2022),
HiFi-Net (Guo et al., 2023), PSCC-Net (Liu et al., 2022), CAT-Net (Kwon et al., 2021), and MVSS-
Net (Dong et al., 2022), all of which are retrained on the MMTD-Set for consistency with the
same training setup. ForDeepFake detection, CADDM (Dong et al., 2023), HiFi-DeepFake (Guo
et al., 2023), RECCE (Cao et al., 2022) and Exposing (Ba et al., 2024) are chosen as comparison
methods. Additionally, to assess theexplanation abilityof FakeShield, we compare it with open-
source M-LLMs such as LLaVA-v1.6-34B (Liu et al., 2024), InternVL2-26B (Chen et al., 2024), and
Qwen2-VL-7B (Wang et al., 2024), as well as the closed-source model GPT-4o (OpenAI, 2023).

Evaluation Metrics:For detection, we report image-level accuracy (ACC) and F1 scores. For lo-
calization, we provide Intersection over Union (IoU) and F1 scores. To evaluate interpretability, we
use Cosine Semantic Similarity (CSS) to assess the similarity between the predicted text and ground
truth text by calculating the cosine similarity between their high-dimensional semantic vectors. For
both detection and localization, a default threshold of 0. 5 is applied unless otherwise specified.

Implementation Details: On the MMTD-Set, we initially fine-tune the M-LLM using LoRA
(rank= 128 , alpha= 256 ), such as LLaVA-v1.5-13B (Liu et al., 2024), while simultaneously training
the Domain Tag Generator with full parameters. The model is trained for 10 epochs on 4 NVIDIA
A100 40G GPUs, with a learning rate of 2 × 10 −^4. Afterward, we fine-tune the Tamper Comprehen-
sion Module and Segment Anything Model (Kirillov et al., 2023) with LoRA (rank= 8 , alpha= 16 ),
training for 24 epochs on the same hardware configuration, with a learning rate of 3 × 10 −^4.

#### 4.2 COMPARISON WITHIMAGEFORGERYDETECTIONMETHOD

To verify the superiority and generalization of our method on the image forgery detection task, we
test the detection accuracy on MMTD-Set (Photoshop, DeepFake, AIGC-Editing). As shown in


```
FakeShield
```
```
Has this image been
forged with? Why? Give
the tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area: 1.The picture has been tampered with. The tampering is located in the upper half of the image, specifically in the sky area above
the crowd and to the right of the central figure who is raising his arms. The content of the tampered area includes three fighter jets flying in a formation.
2.Judgment basis: 1.Lighting: The lighting on the fighter jets does not match the rest of the scene. The overall scene is lit from the front, as
2.indicated by the shadows on the ground and the people, but the jets appear to be lit from a different angle.Edges: The edges around the fighter jets are unnaturally sharp and do not blend seamlessly with the sky, suggesting they have
3.been inserted into the image.Resolution: The fighter jets appear to be of a different resolution compared to the rest of the image, with the jets looking
4.slightly more pixelated, which is a common artifact of image tampering.Perspective relationship: The size and positioning of the fighter jets in relation to the crowd and the rest of the environment
seem unnatural. The jets appear disproportionately large compared to the people and the surroundings, which disturbs the perspective and depth of the image.
5.Shadows: There are no corresponding shadows on the ground from the fighter jets, which would be expected if they were actually present in the scene, given the strong sunlight in the rest of the image.
6.Physical laws: The lack of interaction between the fighter jets and the environment, such as the absence of any disturbance ithe smoke from the ground, suggests that the jets were not originally part of the scene. n
```
```
Fake, √
```
```
Figure 4: Detection, localization and explanation results of our FakeShield.
```
Table 1, our FakeShield almost achieves optimal performance across various tampering and testing
data domains. For example, our method outperforms the second-best method, MVSS-Net, with an
ACC of 0.08 and an F1 of 0.05 on the IMD2020 dataset. Notably, since we introduce the domain-
tag guidance strategy, our method not only achieves excellent performance on the traditional IFDL
benchmark but also generalizes to DeepFake and AIGC tampering, achieving 0.93 and 0.98 de-
tection accuracy, respectively. However, other works lack an effective mechanism to handle data
domain conflicts. Even when trained using the same multi-data domain training set as ours, their
detection accuracy remains insufficient.

```
Table 2: Performance comparison on
DFFD and Seq-DeepFake datasets.
```
```
Method DFFD Seq-DeepFake
ACC F1 ACC F
CADDM 0.52 0.60 0.53 0.
HiFi-DeepFake 0.52 0.64 0.52 0.
Exposing 0.82 0.84 0.71 0.
RECCE 0.92 0.92 0.75 0.
FakeShield 0.98 0.99 0.84 0.
```
Furthermore, we compare our method with some re-
cent DeepFake detection approaches on the DFFD
(FaceApp, FFHQ) and Seq-DeepFake, where Expos-
ing and RECCE were retrained on our datasets. As
reported in Table 2, our approach significantly out-
performs all other methods. Notably, the domain tag
mechanism not only alleviates conflicts across diverse
data domains but also promotes complementarity and
mutual enhancement between them. Although specifi-
cally designed for DeepFake detection, these methods
still underperform compared to our FakeShield.

4.3 COMPARISON WITHM-LLMS

To assess the quality of explanation text generation, we employ the tampered descriptions generated
by GPT-4o as ground truth on the MMTD-Set(Photoshop, DeepFake, AIGC-Editing), using cosine
semantic similarity (CSS) to compare the performance of pre-trained M-LLMs against FakeShield.
The results in Table 3 show that our approach consistently achieves the best performance across
nearly all test sets. For instance, on the DSO, our method attains a CSS score of 0.8873, signifi-
cantly surpassing the second-best result from InternVL2-26B. Figure 4 presents an example of our
explanation. The model identifies the three airplanes in the sky as fake and explains its judgment
based on lighting and edges, providing an accurate manipulation region mask.

Notably, some pre-trained M-LLMs exhibit limited proficiency in detecting tampered content. For
instance, when tampering causes clear physical law violations, these M-LLMs leverage their pre-
training knowledge to make reasonably correct judgments. However, they struggle to perform more
precise analyses, like detecting lighting or perspective inconsistencies, reducing overall accuracy.

Table 3: Comparative results(CSS↑) of the pre-trained M-LLMs and FakeShield in tampering ex-
planation capabilities on the MMTD-Set.

```
Method PhotoShop DeepFake AIGC-Editing
CASIA1+ IMD2020 Columbia Coverage NIST DSO Korus
GPT-4o 0.5183 0.5326 0.5623 0.5518 0.5732 0.5804 0.5549 0.5643 0.
LLaVA-v1.6-34B 0.6457 0.5193 0.5578 0.5655 0.5757 0.5034 0.5387 0.6273 0.
InternVL2-26B 0.6760 0.5750 0.6155 0.6193 0.6458 0.6484 0.6297 0.6570 0.
Qwen2-VL-7B 0.6133 0.5351 0.5603 0.5702 0.5559 0.4887 0.5260 0.6060 0.
FakeShield 0.8758 0.7537 0.8791 0.8684 0.8087 0.8873 0.7941 0.8446 0.
```
#### 4.4 COMPARISON WITHIMAGEFORGERYLOCATIONMETHOD

To assess the model’s capability to locate tampered regions, we conduct comparisons with some
competitive IFDL methods on the MMTD-Set. As present in Table 4, our method consistently


Table 4: Comparative results of tamper localization capabilities between competing IFDL methods
and FakeShield, tested on MMTD-Set(Photoshop, DeepFake, AIGC-Editing).

```
Method CASIA1+ IMD2020 Columbia NIST DSO Korus DeepFake AIGC-Editing
IoU F1 IoU F1 IoU F1 IoU F1 IoU F1 IoU F1 IoU F1 IoU F
SPAN 0.11 0.14 0.09 0.14 0.14 0.20 0.16 0.21 0.14 0.24 0.06 0.10 0.04 0.06 0.09 0.
ManTraNet 0.09 0.13 0.10 0.16 0.04 0.07 0.14 0.20 0.08 0.13 0.02 0.05 0.03 0.05 0.07 0.
OSN 0.47 0.51 0.38 0.47 0.58 0.69 0.25 0.33 0.32 0.45 0.14 0.19 0.11 0.13 0.07 0.
HiFi-Net 0.13 0.18 0.09 0.14 0.06 0.11 0.09 0.13 0.18 0.29 0.01 0.02 0.07 0.11 0.13 0.
PSCC-Net 0.36 0.46 0.22 0.32 0.64 0.74 0.18 0.26 0.22 0.33 0.15 0.22 0.12 0.18 0.10 0.
CAT-Net 0.44 0.51 0.14 0.19 0.08 0.13 0.14 0.19 0.06 0.10 0.04 0.06 0.10 0.15 0.03 0.
MVSS-Net 0.40 0.48 0.23 0.31 0.48 0.61 0.24 0.29 0.23 0.34 0.12 0.17 0.10 0.09 0.18 0.
FakeShield 0.54 0.60 0.50 0.57 0.67 0.75 0.32 0.37 0.48 0.52 0.17 0.20 0.14 0.22 0.18 0.
```
```
Tampered Image
```
```
GT
```
- Mask

```
PSCC
```
- Net

```
OSN
```
```
CAT
```
- Net

```
MVSS
```
- Net

```
FakeShield
```
Figure 5: Comparisons between our FakeShield and other competitive methods. The samples, from
left to right, are drawn from IMD2020, CASIA1+, Columbia, NIST16, Korus, DSO, and DeepFake.

surpasses other methods across almost all test datasets. For instance, on the IMD2020, our method
outperforms the suboptimal method OSN with notable advantages of 0.12 in IoU and 0.1 in F1 score.
On the CASIA1+, we also lead OSN with an IoU of 0.07 and an F1 of 0.09.

Additionally, subjective comparisons of several methods are shown in Figure 5, where our approach
precisely captures the tampered areas, producing clean and complete masks. In contrast, methods
like PSCC-Net exhibit dispersed attention over the image, resulting in blurred segmentations and an
overly broad predicted tampering area. Notably, as our segmentation module MFLM is based on
the pre-trained visual model SAM, it inherits SAM’s powerful semantic segmentation capabilities
and can accurately segment targets with clear semantic information (Columns 1 and 2 of Figure 5).
Additionally, due to the diverse tampering types in MMTD-Set, our method can also accurately
segmenting tampered areas that lack distinct semantic information (Columns 3 of Figure 5).

#### 4.5 ROBUSTNESSSTUDY

With the widespread use of the Internet and social media, individuals are increasingly receiving im-
ages degraded by transmission artifacts such as JPEG compression and Gaussian noise. Our model’s
performance on MMTD-Set(CASIA1+) under these degradations is reported in Table 5, which in-
cludes four common degradation types: JPEG compression qualities of 70 and 80 , and Gaussian


noise variances of 5 and 10. As M-LLMs primarily emphasize high-level semantic information,
although we do not specifically add degraded data during training, FakeShield demonstrates robust-
ness to low-level visual distortions and noise. JPEG compression and Gaussian noise, commonly
associated with social media, have minimal effect on its performance, highlighting the stability,
robustness, and practical advantages of our approach.

```
Table 5: Explanation and location perfor-
mance under different degradations.
```
```
Method Explanation Location
CSS IoU F
JPEG 70 0.8355 0.5022 0.
JPEG 80 0.8511 0.5026 0.
Gaussian 5 0.8283 0.4861 0.
Gaussian 10 0.8293 0.4693 0.
Original 0.8758 0.5432 0.
Figure 6: Ablation study on the DTE-FDM.
```
#### 4.6 ABLATIONSTUDY

Ablation Study on Domain Tag Generator:To validate that the proposed domain tag effectively
mitigates domain discrepancies and enhances the model’s generalization across diverse data do-
mains, we conducted an ablation study. Specifically, we removed the domain tag generator (DTG)
and trained the FakeShield with identical configurations on the MMTD-Set, the test results are dis-
played in Table 6. Without the DTG, the model’s detection performance declined across test sets
from each data domain. Notably, the detection ACC and F1 score for IMD2020 decreased by 0.
and 0.11. This demonstrates that without the support of the DTG module, the model struggles to
effectively differentiate between various data domains, leading to more pronounced data conflicts
and a significant reduction in both generalization and practical applicability.

```
Table 6: Performance comparison of FakeShield
with and without DTG on different datasets.
Method CASIA1+ IMD2020 DeepFake AIGC-Editing
ACC F1 ACC F1 ACC F1 ACC F
Ours w/o DTG 0.92 0.92 0.71 0.79 0.89 0.90 0.72 0.
Ours 0.95 0.95 0.83 0.90 0.98 0.99 0.93 0.
```
Ablation Study on LLM in the DTE-FDM:
Furthermore, to verify the necessity of the
LLM in the DTE-FDM, we design a variant of
FakeShield, which removes the LLM and di-
rectly input{Tins,Ttag,Timg}into the tam-
per comprehension module, adjusting its train-
ing objective to directly produce the description
Odetand the maskMloc. Using the same training configurations, we train the variant and our orig-
inal framework for 25 epochs, evaluating their localization accuracy on the CASIA1+ dataset. As
shown in Figure 6, after removing the LLM, the localization performance consistently lags behind
the original framework throughout the entire training process and it converges earlier. It proves
that joint training detection and localization via a single MFLM tends to cause notable performance
degradation than our decoupled module design, which further highlights the critical role of our LLM
module in enhancing the semantic understanding of the proposed framework.

## 5 CONCLUTION

In this work, we present the first application of an M-LLM for the explanation IFDL, marking a
significant advancement in the field. Our proposed framework, FakeShield, excels in tampering de-
tection while delivering comprehensive explanations and precise localization, demonstrating strong
generalization across a wide range of manipulation types. These features make it a versatile and
practical tool for diverse real-world applications. Looking to the future, this work can play a crucial
role in several areas, such as aiding in the improvement of laws and regulations related to digital
content manipulation, informing the development of guidelines for generative artificial intelligence,
and promoting a clearer and more trustworthy online environment. Additionally, FakeShield can as-
sist in evidence collection for legal proceedings and help correct misinformation in public discourse,
ultimately contributing to the integrity and reliability of digital media.


## REFERENCES

Vishal Asnani, Xi Yin, Tal Hassner, and Xiaoming Liu. Malp: Manipulation localization using a
proactive scheme. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2023.

Zhongjie Ba, Qingyu Liu, Zhenguang Liu, Shuang Wu, Feng Lin, Li Lu, and Kui Ren. Exposing
the deception: Uncovering more forgery clues for deepfake detection. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 38, pp. 719–728, 2024.

Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end
reconstruction-classification learning for face forgery detection. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 4113–4122, 2022.

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua
Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint
arXiv:2311.12793, 2023a.

Xinru Chen, Chengbo Dong, Jiaqi Ji, Juan Cao, and Xirong Li. Image manipulation detection by
multi-view multi-scale supervision. InProceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), 2021.

Yirong Chen, Zhenyu Wang, Xiaofen Xing, Zhipei Xu, Kai Fang, Junhong Wang, Sihang Li,
Jieling Wu, Qi Liu, Xiangmin Xu, et al. Bianque: Balancing the questioning and suggestion
ability of health llms with multi-turn health conversations polished by chatgpt. arXiv preprint
arXiv:2310.15896, 2023b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong
Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 24185–24198, 2024.

Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, and Anil K Jain. On the detection of digital
face manipulation. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
recognition, pp. 5781–5790, 2020.

Tiago Jose De Carvalho, Christian Riess, Elli Angelopoulou, Helio Pedrini, and Anderson ́
de Rezende Rocha. Exposing digital image forgeries by illumination color classification.IEEE
Transactions on Information Forensics and Security, 8(7):1182–1194, 2013.

Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding.
arXiv preprint arXiv:1810.04805, 2018.

Chengbo Dong, Xinru Chen, Ruohan Hu, Juan Cao, and Xirong Li. Mvss-net: Multi-view multi-
scale supervised networks for image manipulation detection.IEEE Transactions on Pattern Anal-
ysis and Machine Intelligence, 45(3):3539–3553, 2022.

Jing Dong, Wei Wang, and Tieniu Tan. Casia image tampering detection evaluation database. In
Proceedings of the IEEE China Summit and International Conference on Signal and Information
Processing (ChinaSIP), 2013.

Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, and Zheng Ge. Implicit identity
leakage: The stumbling block to improving deepfake detection generalization. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3994–4004, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
arXiv preprint arXiv:2407.21783, 2024.

FaceApp Limited. Faceapp.https://www.faceapp.com/, 2017. Accessed: 2024-08-16.

Xiao Guo, Xiaohong Liu, Zhiyuan Ren, Steven Grosz, Iacopo Masi, and Xiaoming Liu. Hierar-
chical fine-grained image forgery detection and localization. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2023.


Edward J Hu, belong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. InInternational Con-
ference on Learning Representations (ICLR), 2022.

Xiaoxiao Hu, Qichao Ying, Zhenxing Qian, Sheng Li, and Xinpeng Zhang. Draw: Defending
camera-shooted raw against image manipulation. InProceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), 2023.

Xuefeng Hu, Zhihan Zhang, Zhenye Jiang, Syomantak Chaudhuri, Zhenheng Yang, and Ram Neva-
tia. Span: Spatial pyramid attention network for image manipulation localization. InProceedings
of the European Conference on Computer Vision (ECCV), 2020.

Zhengchao Huang, Bin Xia, Zicheng Lin, Zhun Mou, and Wenming Yang. Ffaa: Multimodal large
language model based explainable open-world face forgery analysis assistant. arXiv preprint
arXiv:2408.10072, 2024.

Ashraful Islam, Chengjiang Long, Arslan Basharat, and Anthony Hoogs. Doa-gan: Dual-order at-
tentive generative adversarial network for image copy-move forgery detection and localization. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
2020.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv
preprint arXiv:2304.02643, 2023.

Vladimir V Kniaz, Vladimir Knyaz, and Fabio Remondino. The point where reality meets fan-
tasy: Mixed adversarial generators for image splice detection. Advances in neural information
processing systems, 32, 2019.

Paweł Korus and Jiwu Huang. Multi-scale analysis strategies in prnu-based tampering localization.
IEEE Transactions on Information Forensics and Security, 12(4):809–824, 2016.

Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee. Cat-net: Compression ar-
tifact tracing network for detection and localization of image splicing. InProceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Rea-
soning segmentation via large language model. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 9579–9589, 2024.

Haodong Li and Jiwu Huang. Localization of deep inpainting using high-pass fully convolutional
network. InProceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
2019.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-
training for unified vision-language understanding and generation. InInternational Conference
on Machine Learning (ICML), 2022.

Runyi Li, Xuanyu Zhang, Zhipei Xu, Yongbing Zhang, and Jian Zhang. Protect-your-ip: Scalable
source-tracing and attribution against personalized generation.arXiv preprint arXiv:2405.16596,
2024.

Yuanman Li and Jiantao Zhou. Fast and effective image copy-move forgery detection via hierar-
chical feature point matching.IEEE Transactions on Information Forensics and Security, 14(5):
1307–1322, 2018.

Yue Li, Dong Liu, Houqiang Li, Li Li, Zhu Li, and Feng Wu. Learning a convolutional neural
network for image compact-resolution. IEEE Transactions on Image Processing, 28(3):1092–
1107, 2018.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Doll ́ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. InProceedings of
the European Conference on Computer Vision (ECCV), 2014.


Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances
in neural information processing systems, 36, 2024.

Xiaohong Liu, Yaojie Liu, Jun Chen, and Xiaoming Liu. Pscc-net: Progressive spatio-channel
correlation network for image manipulation detection and localization. IEEE Transactions on
Circuits and Systems for Video Technology, 32(11):7505–7517, 2022.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool.
Repaint: Inpainting using denoising diffusion probabilistic models. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Xiaochen Ma, Bo Du, Xianggen Liu, Ahmed Y Al Hammadi, and Jizhe Zhou. Iml-vit: Image
manipulation localization by vision transformer.arXiv preprint arXiv:2307.14863, 2023.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling
drag-style manipulation on diffusion models.arXiv preprint arXiv:2307.02421, 2023.

Tian-Tsong Ng, Jessie Hsu, and Shih-Fu Chang. Columbia image splicing detection evaluation
dataset.DVMM lab. Columbia Univ CalPhotos Digit Libr, 2009.

Yuval Nirkin, Lior Wolf, Yosi Keller, and Tal Hassner. Deepfake detection based on discrepancies
between faces and their context.IEEE Transactions on Pattern Analysis and Machine Intelligence,
44(10):6111–6121, 2021.

Adam Novozamsky, Babak Mahdian, and Stanislav Saic. Imd2020: A large-scale annotated dataset
tailored for detecting manipulated images. InProceedings of the IEEE/CVF Winter Conference
on Applications of Computer Vision Workshops, pp. 71–80, 2020.

NVIDIA Corporation. Flickr-faces-hq dataset (ffhq). https://github.com/NVlabs/
ffhq-dataset. Accessed: 2024-09-30.

R OpenAI. Gpt-4 technical report. arxiv 2303.08774.View in Article, 2(5), 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M ̈uller, Joe
Penna, and Robin Rombach. Sdxl: improving latent diffusion models for high-resolution image
synthesis.arXiv preprint arXiv:2307.01952, 2023.

Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham
Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel
grounding large multimodal model. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 13009–13018, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ̈
resolution image synthesis with latent diffusion models. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR), 2022.

Ronald Salloum, Yuzhuo Ren, and C-C Jay Kuo. Image splicing localization using a multi-task fully
convolutional network (mfcn).Journal of Visual Communication and Image Representation, 51:
201–209, 2018.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine
Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. Multitask prompted training enables
zero-shot task generalization. InInternational Conference on Learning Representations, 2022.

Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and recovering sequential deepfake manipulation.
InEuropean Conference on Computer Vision, pp. 712–728. Springer, 2022.

Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M Jorge Cardoso. Generalised
dice overlap as a deep learning loss function for highly unbalanced segmentations. InDeep
Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support:
Third International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017,
Held in Conjunction with MICCAI 2017, Qu ́ebec City, QC, Canada, September 14, Proceedings
3 , pp. 240–248. Springer, 2017.


Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha,
Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky.
Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu,
Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the
world at any resolution.arXiv preprint arXiv:2409.12191, 2024.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang,
Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv
preprint arXiv:2311.03079, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.

Bihan Wen, Ye Zhu, Ramanathan Subramanian, Tian-Tsong Ng, Xuanjing Shen, and Stefan Win-
kler. Coverage—a novel database for copy-move forgery detection. In2016 IEEE international
conference on image processing (ICIP), pp. 161–165. IEEE, 2016.

Haiwei Wu, Jiantao Zhou, Jinyu Tian, and Jun Liu. Robust image forgery detection over online
social network shared images. InProceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2022.

Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan. Mantra-net: Manipulation tracing net-
work for detection and localization of image forgeries with anomalous features. InProceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Xinyu Yang and Jizhe Zhou. Research about the ability of llm in the tamper-detection area.arXiv
preprint arXiv:2401.13504, 2024.

Qichao Ying, Zhenxing Qian, Hang Zhou, Haisheng Xu, Xinpeng Zhang, and Siyi Li. From image
to imuge: Immunized image generation. InProceedings of the ACM international conference on
Multimedia (MM), 2021.

Qichao Ying, Xiaoxiao Hu, Xiangyu Zhang, Zhenxing Qian, Sheng Li, and Xinpeng Zhang. Rwn:
Robust watermarking network for image cropping localization. InProceedings of the IEEE Inter-
national Conference on Image Processing (ICIP), 2022.

Qichao Ying, Hang Zhou, Zhenxing Qian, Sheng Li, and Xinpeng Zhang. Learning to immunize
images for tamper localization and self-recovery. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2023.

Jiwen Yu, Xuanyu Zhang, Youmin Xu, and Jian Zhang. Cross: Diffusion model makes controllable,
robust and secure image steganography.Advances in Neural Information Processing Systems, 36,
2024a.

Zeqin Yu, Jiangqun Ni, Yuzhen Lin, Haoyi Deng, and Bin Li. Diffforensics: Leveraging diffusion
prior to image forgery detection and localization. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 12765–12774, 2024b.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image
diffusion models. InProceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV), 2023.

Xuanyu Zhang, Runyi Li, Jiwen Yu, Youmin Xu, Weiqi Li, and Jian Zhang. Editguard: Versatile
image watermarking for tamper localization and copyright protection. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11964–11974, 2024a.

Xuanyu Zhang, Jiarui Meng, Runyi Li, Zhipei Xu, Yongbing Zhang, and Jian Zhang. Gs-hider: Hid-
ing messages into 3d gaussian splatting. InAdvances in Neural Information Processing Systems
(NeurIPS), 2024b.


Xuanyu Zhang, Youmin Xu, Runyi Li, Jiwen Yu, Weiqi Li, Zhipei Xu, and Jian Zhang. V2a-mark:
Versatile deep visual-audio watermarking for manipulation localization and copyright protection.
InProceedings of the ACM international conference on Multimedia (MM), 2024c.

Yue Zhang, Ben Colman, Ali Shahriyari, and Gaurav Bharaj. Common sense reasoning for deep
fake detection.arXiv preprint arXiv:2402.00126, 2024d.

Xinshan Zhu, Yongjun Qian, Xianfeng Zhao, Biao Sun, and Ya Sun. A deep learning approach to
patch-based image inpainting forensics. Signal Processing: Image Communication, 67:90–99,
2018.


## A APPENDIX

#### A.1 LIMITATIONS ANDFUTUREWORKS

One limitation of our current framework is its suboptimal performance when handling more complex
types of deepfake tampering, such as identity switching and full-face generation. These types of
manipulations introduce unique challenges that our model currently struggles to address effectively.
To address this, future work will focus on several key optimizations. First, we plan to incorporate a
Chain-of-Thought (CoT) (Wei et al., 2022) mechanism to enhance the model’s reasoning abilities,
enabling it to detect more subtle manipulations in deepfake content. Second, we will expand our
training dataset to include a broader range of deepfake samples, encompassing various tampering
techniques and scenarios, to improve the model’s generalization. Finally, we will optimize specific
modules within the framework to better handle these complex tampering types, creating a more
robust and adaptable detection system. These improvements are expected to significantly enhance
the framework’s performance across a wider spectrum of deepfake domains.

#### A.2 DATASOURCES

We collected source data from the dataset mentioned in Section 4.1, with the details provided in
Table 7.

It is noted that the FaceApp and FFHQ datasets are part of the DFFD (Dang et al., 2020) dataset.
We follow their original configuration to divide the training and validation sets. FFHQ (NVIDIA
Corporation) consists of real face images, while FaceApp (FaceApp Limited, 2017) contains fake
faces generated by the FaceApp (FaceApp Limited, 2017) tool, which manipulates facial attributes
in the images.

Regarding the construction process of the AIGC-Editing dataset, we first collected 20,000 real im-
ages from the COCO (Lin et al., 2014) dataset and used the SAM (Kirillov et al., 2023) tool to
segment the masks of all targets. We then selected the target mask with the third-largest area in
each image and applied the Stable-Diffusion-Inpainting (Lugmayr et al., 2022) method to partially
redraw this section. To ensure a balance between positive and negative samples, we further extracted
an additional 20,000 images from the COCO dataset, distinct from the previously selected ones.

```
Table 7: Summary of datasets used for training, and evaluation.
Dataset Real Fake Copy-Move Splicing Removal DeepFake AIGC-Edit
#Training
Fantastic Reality 16,592 19,423 - 19,423 - - -
CASIAv2 7,491 5,123 3,295 1,828 - - -
FFHQ 10000 - - - - - -
FaceAPP - 7,308 - - - 7,308 -
COCO 20,000 - - - - - -
AIGC-Editing - 20,000 - - - - 20,
#Evaluation
CASIAv1+ 800 920 459 461 - - -
Columbia - 180 - 180 - - -
Coverage - 100 100 - - - -
NIST - 564 68 288 208 - -
IMD2020 414 2,010 - 2,010 - - -
DSO - 100 - 100 - - -
Korus - 220 - 220 - - -
FFHQ 1,000 - - - - - -
FaceAPP - 1,000 - - - 1,000 -
COCO 20,000 - - - - - -
AIGC-Editing - 20,000 - - - - 20,
```

#### A.3 MOREABLATIONEXPERIMENTS

Exploring the Introduction of an Error-Correction Mechanism in MFLM: We observe that our
MFLM inherently possesses some error-correction capability. This is because, during training, we
use potentially inaccurateOdetas input and encourage the model to outputMˆloc. Thus, during
testing, ifOdetcontains minor inaccuracies, such as incorrect descriptions of tampered locations,
MFLM can correct them. To further explore the impact of incorporating an error correction mech-
anism during MFLM training on localization accuracy, we conducted an ablation study. During

MFLM training, we used the ground truthOˆdetto constrain TCM’s output to correct the inputOdet.
The results in the Table 8 indicate that correctingOdetdoes not improveMlocpredictions, possibly
due to interference between mask and text optimization.

```
Table 8: Exploration on introducing error correction in MFLM on localization performance.
```
### Method

### CASIA1+ IMD2020 DeepFake AIGC-Editing

### IoU F1 IoU F1 IoU F1 IoU F

### Using correctOdet 0.51 0.56 0.49 0.55 0.13 0.21 0.12 0.

### FakeShield 0.54 0.60 0.50 0.57 0.14 0.22 0.18 0.

More Ablation Study on LLM in the DTE-FDM: To investigate the roles ofTtag,Timg, and
Tins. When training MFLM, we kept other training configurations unchanged and adjusted the
inputs to: {Tins,Timg}and{Tins,Ttag}. The test results are shown in Table 9. It is evident
that using input{Tins,Timg}is slightly better than{Tins,Ttag}, but both are inferior to our
method with the input{Odet,Timg}.This indicates that using an LLM to first analyze the basis for
tampering and then guiding the localization achieves better results.

```
Table 9: Ablation Study on LLM in the DTE-FDM with different input combinations.
```
## Method

## CASIA1+ IMD2020 DeepFake AIGC-Editing

## IoU F1 IoU F1 IoU F1 IoU F

## Tins,Timg 0.50 0.55 0.48 0.53 0.13 0.21 0.12 0.

## Tins,Ttag 0.49 0.54 0.47 0.52 0.12 0.19 0.12 0.

## Tins,Ttag,Timg 0.51 0.55 0.48 0.54 0.13 0.2 0.11 0.

## Odet,Timg(Ours) 0.54 0.60 0.50 0.57 0.14 0.22 0.18 0.

#### A.4 MOREAIGC-EDITING GENERALIZATION EXPERIMENTS

To further verify the model’s generalization performance in the AIGC-Editing data domain, we
expanded our test set by constructing 2,000 tampered images using controlnet inpainting (Zhang
et al., 2023) and SDXL inpainting (Podell et al., 2023), which the model had not encountered before.
We selected MVSS-Net, CAT-Net, and HiFi-Net, which performed well in the AIGC-Editing data
domain in Tables 1, as comparison methods for testing. The detection and localization test results
are presented in Table 10. Our method demonstrates leading performance on unseen datasets and
exhibits strong generalization capabilities.

```
Table 10: Generalized performance comparison on ControlNet and SDXL Inpainting datasets.
```
```
Method
```
```
ControlNet Inpainting SDXL Inpainting
ACC Image-level F1 IoU Pixel-level F1 ACC Image-level F1 IoU Pixel-level F
MVSS-Net 0.38 0.27 0.05 0.09 0.35 0.20 0.05 0.
CAT-Net 0.90 0.89 0.11 0.17 0.86 0.83 0.06 0.
HiFi-Net 0.49 0.66 0.17 0.28 0.44 0.61 0.02 0.
Ours 0.99 0.99 0.18 0.24 0.99 0.99 0.18 0.
```

#### A.5 ANSWERANALYSIS

Figure 7 presents the adjective and noun word clouds for the ground truth (GT) descriptions in the
MMTD-Set and the answer descriptions generated by our FakeShield. It is evident that, through
effective fine-tuning, FakeShield can be guided by the dataset to assess both image-level semantic
plausibility (e.g., ”physical law,” ”texture”) and pixel-level tampering artifacts (e.g., ”edge,” ”reso-
lution”) to determine whether an image is real.

```
(a) MMTD-Set (b) FakeShield Generated Answer
```
Figure 7:The noun and verb word clouds in the MMTD-Set and the FakeShield Generated
Answer.It can be seen that the high-frequency vocabulary of the two is basically the same.

A.6 PROMPTS

During the process of using GPT-4o to construct the MMTD-Set, we meticulously designed distinct
prompts for each category of tampered data to guide GPT-4o in focusing on specific aspects for
image analysis, as illustrated in Figure 8 and Figure 9.

#### A.7 EXAMPLES

Comparison of subjective results of mainstream M-LLM: As mentioned in Section 4.3, we se-
lected some M-LLM output samples, as shown in Figures 10 and 11.

FakeShield output subjective samples: We selected several results from FakeShield’s testing on
PhotoShop, DeepFake, and AIGC-Editing datasets, as displayed in Figures 12, 13, 14, 15, 16 and 17.

MMTD-Set data set example: We select some samples from the MMTD-Set data set and display
them in Figures 18, 19, and 20.


```
You are an AI visual assistant that can help humans analyze some tampered images. You will receive two images, the first is the tampered image A, and
the second is the binary mask image B of the tampered region (a value of 1 (white) indicates the tampered area, and a value of 0 (black) indicates the
untampered area).
Now your task is to use the binary mask provided for the tampered picture A and the tampered area of the picture B:
```
1. Describe the location of the tampering area in the diagram,
2. Describe in detail the contents of the tampered area,
3. Describe visible details in the image that have been tampered with
In the answer, don't directly mention the binary mask. Always assume that you are simply observing tampered images
When describing Problem 1, use natural language and give both the relative and absolute position of the tampered area, but don't give an ambiguous,
unclear description:
(1) Relative position: Use the relative position between objects in the picture, such as above the crowd, on the wall, in the sky, on the table, etc.;
(2) Absolute position: The tampering area is relative to the position of the entire image, such as the left side, right side, top half, bottom half, bottom left
corner, lower right corner, upper left corner, upper right corner, etc
When describing Problem 2, please use natural language to describe in detail the types of objects, the number of objects, the actions of objects, and the
properties of objects in the tampering area selected by the mask, but do not give ambiguous or unclear descriptions.
When describing Problem 3, consider the visible details caused by tampering from these perspectives, but don't give an ambiguous, unclear description,
or something is challenge:
(1) Lighting: Please carefully observe the overall style, color and details of the picture to determine whether there are visual inconsistencies. Pay special
attention to the consistency of light, shadows, and colors, and whether there are any unnatural areas or marks. If there are multiple objects or people in
the picture, they should be illuminated by the same light and the shadows should be consistent. Inconsistent lighting and shadowing may suggest image
compositing or modification.
(2) Copy and paste: Observe whether there are some duplicate areas or blocks in the image, which may be evidence left by the copy and paste operation.
(3) Edges: Check whether there are any unnatural pixel distributions or edges in the image. Particular attention is paid to the presence of discontinuous or
inconsistent edges, and the presence of visible shear or synthesis marks.
(4) Resolution: Please check the resolution and compression traces of the picture. Composite or edited images may show unnatural pixel blur, jaggedness,
or excessive compression.
(5) Perspective relationship: Observe the perspective and proportion relationship in the picture. Real photos should be consistent in perspective and
proportions, and if there are unusual or unnatural perspective relationships, it could be a sign of compositing or editing. Check whether the image has a
reasonable change in depth of field, that is, whether the degree of bokeh between the foreground, background, and subject conforms to the actual laws of
physics.
(6) Shadows: Observe whether there are reasonable reflections and shadows in the photo. Realistic photos often produce reflections and shadows based
on the light source, while composite photos may have unnatural shadows or reflections
(7) Text: If the photo contains words or logos, you can check whether it is clear and legible and consistent with the surrounding environment.
Unnaturalness or incongruity of text or logos may suggest tampering with the photograph.
(8) Physical laws: Check whether the content of the picture violates the physical laws, such as the movement trajectory of the object, the position and
shape of the reflection, etc.

```
You are an AI visual assistant that helps humans analyze some real images that have not suffered tampering. You will receive an image that is a real
image that has not been tampered with
Now your task is to use the provided real image that has not been tampered with A: 1. Describe the angles from which this image can be seen that he has
not been tampered with, and can be seen that this is a real and trustworthy real image obtained directly from the camera, 2. Please focus on describing
some of the visible details that are consistent with the real scenario, and the real laws of physics, and 3. If you were put in a pile of tampered images, and
were asked to judge If you were to judge whether this picture has been tampered with, from what angle would you analyze and judge it and give detailed
reasons?
In answering the above questions, consider the visible details triggered by tampering from these perspectives:
(1) Lighting: Please take a close look at the overall style, color and details of the picture to determine whether there is any visual inconsistency. Please
pay particular attention to the consistency of lighting, shadows and colors, as well as the presence of unnatural areas or marks. If there are multiple
objects or people in the picture, they should be illuminated by the same light and the shadows should be consistent. Inconsistent lighting and shadows
may suggest image compositing or modification.
(2) Copy and Paste: Look for signs of duplication or mirroring in the image, which may be evidence left by a copy and paste operation.
(3) Edges: Check for unnatural pixel distribution or edges in the image. Pay particular attention to the presence of discontinuous or inconsistent edges, as
well as obvious traces of clipping or compositing.
(4) Resolution: Please check the resolution and traces of compression in the image. Composited or edited images may show unnatural pixel blurring,
jaggedness, or excessive compression marks.
(5) Perspective: Observe the perspective and scale relationships in the picture. Real photographs should be consistent in terms of perspective and
proportions. If abnormal or unnatural perspective relationships appear, it may be a sign of compositing or editing. Look for reasonable changes in depth
of field in the image, i.e., whether the degree of vignetting between the foreground, background and subject is consistent with actual physical laws.
(6) Shadows: Look for reasonable reflections and shadows in the photograph. Real photographs usually produce reflections and shadows accordingly to
the light source, while composite photographs may have unnatural shadows or reflections
(7) Text: If a photograph contains text or logos, check that they are legible and in keeping with their surroundings. Unnatural or incongruous text or logos
may suggest photo tampering.
(8) Physical laws: Check whether the content of the picture violates physical laws, such as the trajectory of the object's movement, the position and shape
of the reflection, and so on.
```
```
(a) Prompt for PhotoShop&AIGC tampered images
```
```
(b) Prompt for authentic scene images
```
Figure 8:Prompts for GPT-4o when constructing MMTD-Set.(a) Analysis guide prompt de-
signed for PhotoShop tampered and AIGC-Editing tampered images. (b) Analysis guidance prompt
designed for real scene images.


```
You are an artificial intelligence visual assistant that helps humans analyze some real face images that have not been tampered with by deepfake. You will receive
an image of a real face that has not been tampered with by deepfake
Now, your task is to use the provided image of a real face that has not been tampered with by deepfake Answer: 1. Describe the angles from which you can see that
this image has not been tampered with by deepfake, and that you can see that this is a real and believable image of a real face straight from the camera, 2. Focus on
describing some of the visible details that are consistent with a real face and the real laws of physics, and 3. What would happen if you were put placed in a pile of
images that have been tampered with by deepfake and asked to determine whether this image has been tampered with, which perspective would you analyze the
judgment from and give detailed reasons?
When describing parts of the human face, common descriptions include
head, face, forehead, eyes, eyebrows, eyelids, eyelashes, nose, nostrils, cheeks, ears, mouth, lips, chin, jaw, hair, neck, eyeglasses, earrings, headphones, scalp,
hairline, sideburns, temples, crown, back of the neck, jawline, teeth, tongue, gums, mustache, beard, headband, hat, headdress, headscarf, veil, hairpiece, hairbrush,
headband
In answering the above questions, consider the visible details triggered by deepfake tampering from the following perspectives:
```
1. symmetrical facial features: Deepfake-generated faces may exhibit unnatural perfect symmetry, lacking the subtle asymmetries common in real faces.
2. Blurred or distorted edges: Deepfake processing may result in blurred or distorted edges on processed faces, especially when the face is digitally superimposed
on another body.
3. Inconsistent Lighting and Shadows: The Deepfake algorithm may have difficulty accurately matching the lighting and shadows in the original image, resulting
in differences in the direction or intensity of lighting across the face.
4. Unnatural facial expressions: Deepfake may produce facial expressions that look unnatural, exaggerated, or out of sync with the rest of the image.
5. Disproportionate Facial Proportions: Deepfake operations may result in facial proportions that do not correspond to the gender or age of the character, such as a
man's face on a woman's body, or vice versa.
6. Inconsistent skin texture and tone: Deepfake-generated faces may exhibit unnatural skin texture or tone, such as overly smooth or pixelated skin that is different
from the surrounding area.
7. Missing or inconsistent eye reflections: Deepfake operations may result in missing or inconsistent eye reflections, which may provide clues to the realism of the
image.
8. Altered hairstyles: Some Deepfake algorithms only tamper with hair, and may change hair color and hairstyle, or add long hair for boys and short hair for girls.
9. Makeup irregularities: Deep forgery algorithms may introduce makeup styles that are inconsistent with an individual's gender or age, or exhibit poor makeup
quality.
10. Contextual inconsistencies: Images generated by Deepfake may contain inconsistencies with the overall context of the image, such as differences in
perspective, clothing, or surroundings, suggesting manipulation of the image.
11. Unreasonable accessories: Some deepfake algorithms add glasses, earrings, hats, masks, etc. to an image, and the edges, lighting relationships, and perspective
of these accessories may be incorrect.

```
You are an AI visual assistant that helps humans analyze some images of human faces that have been tampered with by deepfake. You will receive two images, the
first is the image A of the face tampered by deepfake and the second is the binary mask image B of the tampered area (a value of 1 (white) indicates the tampered
area and a value of 0 (black) indicates the untampered area).
Now, your task is to use the binary masks provided for the tampered regions of the deepfake tampered image A and image B:
```
1. describe the location of the tampered area in the picture,
2. describe in detail the content of the tampered areas,
3. describes the visible details of the tampered area in the image
In your answer, do not refer directly to the binary mask. Always assume that you are just looking at the tampered image
When describing Problem 1, use natural language and give the location of the tampered area, paying particular attention to the fact that if the tampered area is
discontinuous, several larger tampered areas should be described. However, do not give ambiguous and unclear descriptions. Common descriptions of tampered
areas include:
Head, Face, Forehead, Eyes, Eyebrows, Eyelids, Eyelashes, Nose, Nostrils, Cheeks, Ears, Mouth, Lips, Chin, Jaw, Hair, Neck, Glasses, Earrings, Headphones,
Scalp, Hairline, Sideburns, Temple, Crown, Nape, Jawline, Teeth, Tongue, Gums, Moustache, Beard, Headband, Hat, Tiara, Bandana, Veil, Fascinator, Hairpin,
Headscarf
When describing Problem 2, use natural language to describe in detail the types of objects, properties of objects, number of objects, etc. in the tampered area
selected by mask, paying special attention to the fact that, if the tampered area is discontinuous, give a description of the tampered areas that are larger in size. But
do not give ambiguous or unclear descriptions.
When describing Questions 2 and 3, consider the visible details caused by tampering from these perspectives, but do not give an ambiguous, unclear description that
is otherwise challenging:
1. Symmetrical Facial Features: Deepfake-generated faces may exhibit unnaturally perfect symmetry, lacking the subtle asymmetry typically found in real faces.
2. Blur or Distortion Around Edges: Deepfake manipulation may introduce blur or distortion around the edges of the face where the manipulation has taken place,
especially if the face has been digitally overlaid onto another body.
3. Inconsistent Lighting and Shadows: Deepfake algorithms may struggle to accurately match the lighting and shadows in the original image, leading to
discrepancies in lighting direction or intensity across the face.
4. Unnatural Facial Expressions: Deepfakes may produce facial expressions that appear unnatural, exaggerated, or out of sync with the rest of the image.
5. Mismatched Facial Proportions: Deepfake manipulation may result in facial proportions that are inconsistent with the person's gender or age, such as a man's face
on a woman's body or vice versa.
6. Inconsistent Skin Texture and Tone: Deepfake-generated faces may exhibit unnatural skin texture or tone, such as overly smooth or pixelated skin, that differs
from the surrounding areas.
7. Missing or Inconsistent Eye Reflections: Deepfake manipulation may result in missing or inconsistent reflections in the eyes, which can provide clues about the
authenticity of the image.
8. Change hairstyle: Some deepfake algorithms only tamper with hair, and may change hair color and hairstyle, or add long hair for boys and short hair for girls.
9. Irregularities in Makeup Application: Deepfake manipulation may introduce makeup styles that are inconsistent with the person's gender or age, or exhibit poor
application quality.
10. Contextual Inconsistencies: Deepfake-generated images may contain inconsistencies in the overall context of the image, such as discrepancies in perspective,
clothing, or surroundings, that suggest manipulation.
11. Unreasonable accessories: Some deepfake algorithms add glasses, earrings, hats, masks, etc. to the image, and the edges, lighting relationships, and perspective
of these accessories may be incorrect.
12. If there are glasses or sunglasses in the picture, please pay special attention to whether the glasses or sunglasses have been tampered with or not, the rims,
frames, temples, lenses, etc. of the glasses are very susceptible to imperfections and problems.
    (a) Prompt for DeepFake tampered images

```
(b) Prompt for authentic face images
```
Figure 9:More prompts for GPT-4o when constructing MMTD-Set.(a) Analysis guide prompt
designed for DeepFake tampered images. (b) Analysis guidance prompt designed for real face
images.


```
GPT-4o
```
```
InternVL
```
```
LLaVA
```
```
Qwen
```
```
FakeShield
```
```
The image you've provided appears to be of a pastoral scene with cows grazing in a field. However, without access to the original file and
metadata, it's challenging to definitively determine if the image has been edited or remains unprocessed directly from a cameA closer inspection for signs of editing involves analyzing elements such as the edges of the objects, shadows, lighting consistency, and ra.
perspective. In this image, upon zooming in and examining closely, we should look for any unnatural edges that might suggest the cows or
other elements were digitally inserted or altered. We would also check for inconsistent shadows or lighting that doesn't matcdirection of light in the rest of the image, as well as any perspective discrepancies that could indicate manipulation. h the
From the provided image, it's not possible to perform this level of detailed analysis as it would require higher resolution acapabilities. However, if you have suspicions about an image, these are some of the areas you could scrutinize using image ednd itizoom ng software
that allows for closer inspection. If any inconsistencies are found in these areas, they could suggest the image may have beealtered. n digitally
```
```
Upon close inspection, the image appears to be a photograph that has not been significantly tampered with using artificial phmodification techniques such as Photoshop (PS). Here are the details that support this conclusion: oto
```
1. Edges and Perspective Relationships: The edges of the objects, such as the cows and the trees, appear natural. There are no abrupt or
unnatural edges that would suggest digital manipulation. The perspective relationships between the cows, the trees, and the hconsistent and realistic. orizon are
2. Incorrect Semantics: The image depicts a typical pastoral scene with cows grazing in a field. There are no elements that seemout of
place or that would indicate artificial modification. The cows are behaving naturally, and the environment looks authentic. 3. Unnatural Lighting and Darkness: The lighting in the image is consistent and natural. The shadows cast by the cows and thetrees are
proportionate and realistic. There are no areas of the image that appear unnaturally bright or dark, which would be indicativmanipulation. e of photo
4. Color and Detail: The colors in the image are vibrant and natural, with the autumn foliage showing a range of colors from yellow. There are no signs of color banding or unnatural color saturation that would suggest digital alteration. green to red and
5. Background and Foreground: The background and foreground elements blend seamlessly. The transition from the green field tothe trees
and the sky is smooth and natural. There are no signs of blending artifacts or inconsistencies that would indicate photo maniIn conclusion, the image appears to be a genuine photograph with no significant signs of artificial photo modification. The detapulils, lighting, ation.
and overall composition all support this conclusion.

```
I'm unable to provide a definitive answer on whether the photo has been taken directly from the camera or processed using artificial
photo modification techniques. However, I can analyze the image for any potential signs of manipulation.Upon zooming in and examining the edges of the objects, there are no immediately apparent unnatural edges or perspective relationships
that would suggest the photo has been tampered with. The objects in the image, such as the cows and the trees, have a consistentlevel of
detail and do not exhibit any obvious signs of being superimposed or altered.grazing in a field with a backdrop of trees, which is a common and natural scene. There are no elements in the image that see\n\nRegarding semantics, the image depicts a group m oof cows ut of place
or incorrect semantically.As for lighting and darkness, the image shows a natural distribution of light and shadow, with the cows and the field being well-lit,
suggesting that the photo was taken during the day under clear skies. There are no areas of the image that appear overly darkan unnatural way. orbright in
Overall, while it's impossible to say with absolute certainty whether the photo has been modified, there are no obvious signsoftampering
visible upon close inspection. It appears to be a typical, unaltered photograph of cows in a field
```
```
The image appears to be a natural photograph of cows grazing in a field with trees in the background. There are no obvious simanipulation or artificial photo modification techniques such as Photoshop (PS). The edges of the objects, including the cowsgnsand the trees, of digital
appear natural and consistent with what one would expect in a real-world setting.\n\nHere are some observations:\n\n1. **Edges and
Perspective**: The edges of the cows and trees look smooth and natural, without any noticeable artifacts or unnatural lines.and Shadows**: The lighting appears natural, with shadows cast in a way that is consistent with the position of the sun and t\he n2.time of **Lighting
day.\n3. **Colors and Textures**: The colors of the cows and the grass are realistic, and the textures of the animals and the landscape are
consistent with what one would expect in a natural setting.modification techniques. The image looks like a straightforward photograph taken with a camera.\n\nOverall, there are no apparent signs of digital manipulation or artificial photo
```
```
1.Whether the picture has been tampered with / Description of the tampered area: The picture has been tampered with , the tampered
area is located in the lower left quadrant of the image, occupying a central position within this quadrant. It features a cowto be grazing on the grass. that appears
2.Judgment basis : The cow in the tampered area has several visual inconsistencies.
1.2.Firstly, Secondly, the lighting the edges on the cow does not match the rest of the scene; it appears to be under a different light source. around the cow are unnaturally sharp and do not blend seamlessly with the grass background, indicating a
3.possible cutAdditionally, -and the resolution - paste manipulation. of the cow seems to be slightly different from the surrounding environment, with the cow appearing
```
**4.** slightly more blurred than the rest of the image. **The perspective** is also off, as the cow's size and angle do not quite fit with the rest of the scene, making it look out of place.
**5.Shadows** cast by the cow do not conform to the direction of light in the rest of the image, which suggests that the cow was not
    originally part of this scene.

```
Was this photo taken directly from the camera
without any processing? Has it been tampered with
by any artificial photo modification techniques such
as ps? Please zoom in on any details in the image,
paying special attention to the edges of the objects,
capturing some unnatural edges and perspective
relationships, some incorrect semantics, unnatural lighting and darkness etc
```
```
User
```
```
(Real, ×)
```
```
(Fake, √)
```
```
(Real, ×)
```
```
(Real, ×)
```
```
(Real, ×)
```
Figure 10:The response of mainstream pre-trained M-LLM and FakeShield to tampered pic-
tures.This is a sample from the CASIA1+ (Dong et al., 2013) test set, which has been tampered
with by Splicing. The black animal at the bottom of the picture is pasted later. It can be seen that
only FakeShield made the correct judgment and accurately described the tampered area.


```
GPT-4o
```
```
InternVL
```
```
LLaVA
```
```
Qwen
```
```
FakeShield
```
```
The image appears digitally altered, with several signs suggesting manipulation using photo editing software:
```
1. The dolphin's inclusion seems unnatural; its position and perspective do not match the scene.2. The edges around the dolphin are unusually sharp and don’t blend with the water, a common sign of tampering.
3. The lighting on the dolphin is inconsistent with the rest of the scene, indicating it may have been inserted from a differ4. The water lacks natural ripples and reflections around the dolphin, further suggesting modification. ent environment.
These signs indicate the image was likely edited to include the dolphin.

```
Upon examining the image closely, there are several indicators that suggest it may have been tampered with using artificial photo
modification techniques such as Photoshop (PS):1. Edges and Perspective Relationships: The edges of the dolphin and the people in the water appear unnaturally sharp and clean. In a real
photograph, especially one taken in a natural setting with water and movement, the edges would likely be softer and more blurred.\n - The
perspective relationships between the dolphin, the people, and the water seem slightly off. The dolphin appears to be floatindoesn't quite match the natural buoyancy and movement of water. g in a way that
```
2. Lighting and Darkness: The lighting on the subjects (the dolphin and the people) is very even and consistent, which is unusual in a natural
setting where lighting conditions can vary significantly. There are no visible reflections or refractions in the water that win a real photograph. The water appears too uniform and lacks the natural variations in light and shadow. ould be expected
3. Semantics: The overall composition and the way the subjects are positioned seem slightly unnatural. The dolphin appears toaligned with the people, which is unlikely in a spontaneous, candid shot. be perfectly
4. Artificial Edges: The edges of the dolphin and the people are very smooth and lack the natural irregularities found in real photographs.
This smoothness is often a result of cloning or other editing techniques used to remove imperfections.In conclusion, the image shows several signs that it may have been tampered with using photo editing software like Photoshop. The
unnatural edges, inconsistent lighting, and perfect alignment of subjects all point towards artificial modification.

```
I'm unable to provide a direct analysis of the image's authenticity or whether it has been modified. However, I can describe the image: the content of
In the image, there are two individuals in the water, one of whom appears to be swimming with a dolphin. The person is wearinand has a surprised or excited expression on their face. The dolphin is swimming close to the person, and both are in the foregrg aound of the life jacket
image. The water is clear and blue, suggesting a tropical or coastal location. There are no obvious signs of tampering or artmodification visible in the image. The edges of the objects, such as the person's life jacket and the dolphin's body, appear ifinatcial ural and well-
defined. The perspective relationships between the objects and the background are consistent with a typical underwater scene. The
lighting and shadows in the image seem to be consistent with the environment, and there are no obvious incorrect semantics orelements that would suggest the image has been altered. If you have specific concerns about the image's authenticity or if you'r une looking natural
for a detailed analysis of the image's content, I'd be happy to provide more information.
```
```
The image has been tampered with using artificial photo modification techniques, such as Photoshop (PS). The edges of certain objects,
including the dolphin and the people, appear unnaturally smooth and inconsistent with what one would expect in a realare unnatural edges and perspective relationships that suggest digital manipulation. Additionally, the lighting and shadows d-life scenao not align rio. There
naturally, further indicating that the image has undergone alterations.
```
```
1.Description of the Area:The image seems unaltered in the central region, where the dolphin's head and upper body appear naturally integrated into the scene.
2.Judgment Basis: There are several factors supporting this assessment: 1.Lighting: The lighting on the dolphin's head matches the rest of the scene, appearing consistent with the natural light source in the
2.image.Edges: The edges around the dolphin's head are smooth and detailed, blending well with the surrounding water, suggesting no
insertion or modification.
3.Shadows: The shadowing on the dolphin's head aligns with the lighting and shadows present in the rest of the image, indicatinbelongs naturally in the scene. g it
4.Perspective Relationship: The size and position of the dolphin's head seem proportionate relative to its body and the surrounenvironment, with no signs of manipulation. ding
5.Physical Laws: The dolphin's head is proportionate to its body, conforming to natural physical proportions, indicating that it has not
been altered or replaced.
```
```
Was this photo taken directly from the camera
without any processing? Has it been tampered with
by any artificial photo modification techniques such
as ps? Please zoom in on any details in the image,
paying special attention to the edges of the objects,
capturing some unnatural edges and perspective
relationships, some incorrect semantics, unnatural
lighting and darkness etc
```
```
User
```
```
(Fake, ×)
```
```
(Real, √)
```
```
(Fake, ×)
```
```
(Fake, ×)
```
```
(Fake, ×)
```
Figure 11:The response of mainstream pre-trained M-LLM and FakeShield for real images.
This is a sample from the IMD2020 (Novozamsky et al., 2020) test set, which is a real picture. It
can be seen that only FakeShield makes the correct judgment, the rest of the methods consider the
image to be tampered with.


```
FakeShield
```
```
Was this picture taken directly by the camera?
State your conclusion and give reasons. If it is
tampered with, please output the mask of the
tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, in the center of the image, slightly towards the upper half. The tampered area contains a single yellow flower with six petals and a
central corolla, surrounded by a pattern of red leaves with yellow veins.
2.Judgment basis: The tampered area can be identified by the following details:1.Lighting: The lighting on the yellow flower does not match the surrounding red leaves.
The flower appears to have a different light source, suggesting it was inserted from another image.
2.Edges: The edges around the yellow flower are unnaturally sharp and do not blend
3.seamlessly with the surrounding leaves, indicating a possible cutResolution: The yellow flower seems to be of a different resolution compared to the -and-paste manipulation.
surrounding leaves, with the flower appearing slightly more blurred, which is a common artifact of image tampering.
4.Shadows: There are no corresponding shadows under the yellow flower that match the
lighting of the rest of the image, which would be expected if the flower was originally part of the scene.
5.Perspective relationship: The size and angle of the yellow flower do not fully conform
6.to the perspective of the surrounding leaves, making it look out of place.Physical laws: The absence of a shadow under the flower defies the physical law of
light, which would naturally produce a shadow on the leaves below.
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly by
the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, the tampered area is centrally located in the image, occupying a significant portion of the middle ground. It features a
penguin standing on a tatami mat floor, which is not a typical setting for a
2.Judgment basis: penguin, suggesting that the penguin has been digitally inserted into the scene.
1.The lighting on the penguin does not match the rest of the scene; it appears to be lit from the front, while the rest of the environment suggests a more
diffused and ambient light source.
2.The edges around the penguin are also unnaturally sharp and do not blend seamlessly with the tatami mat, indicating a possible cut-and-paste
3.manipulation. The perspective is off as well; the penguin's size and positioning do not
conform to the spatial arrangement of the room, making it appear out of place.
4.Shadows are not consistent either; the penguin casts no shadow on the tatami mat, which defies the physical laws of lighting and shadow, considering the
clear shadows present in the rest of the room.
5.The resolution around the penguin seems to differ from the rest of the image, with the penguin appearing slightly more pixelated, which could be a sign of a
lower resolution image being inserted into a higher resolution scene.
```
```
Fake, √
IoU=0.98
```
```
Fake, √
IoU=0.99
```
Figure 12:Dialogue examples of FakeShield when receiving PhotoShop tampered images.The
first image comes from the CASIA1+ (Dong et al., 2013) dataset and has been tampered with by
Copy-Move. The yellow petals in the center of the picture are the parts pasted later. The second
picture comes from the CASIA1+ (Dong et al., 2013) dataset and has been tampered with by
Splicing. The penguin in the center of the picture is the part pasted later.


```
FakeShield
```
```
Was this picture taken directly by the camera?
State your conclusion and give reasons. If it is
tampered with, please output the mask of the
tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area: 1.The picture has been tampered with, the tampered area is centrally located in the image,
occupying a large, roughly oval shape in the middle of the frame. It appears to be an artificially inserted object or distortion that obscures the view of what would be behind it,
possibly a window or a piece of furniture.
2.Judgment basis: The tampering is evident due to the following observations:1.Lighting: The tampered area has a uniform color and lacks the texture and lighting
2.consistency expected from the rest of the image.Edges: The edges of the tampered area are unnaturally smooth and do not blend with the
surrounding environment, indicating a possible cut-and-paste manipulation.
3.Resolution: The tampered area has a different resolution compared to the rest of the image, appearing blurrier and lacking detail, which suggests it has been digitally altered or
4.inserted.Perspective relationship: The tampered area does not conform to the perspective of the
room; it seems flat and lacks the depth that would be expected from an actual object in
5.that space.Shadows: There are no shadows or reflections that would be consistent with the lighting
in the room, which further suggests that the area has been artificially added or modified.
6.Physical laws: The tampered area does not follow the physical laws of the environment; it appears to be a two-dimensional object in a three-dimensional space without any
interaction with the surroundings.
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly by
the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, specifically in the central portion of the image, extending from the middle to the right side. The tampered area includes the second
2.Judgment basis: The tampering is evident due to several factors:racing car and part of the track behind it.
1.Lighting: The lighting on the tampered car does not match the rest of the scene. It
appears to be under a different light source, as the shadows and highlights do not align with the natural lighting in the rest of the image.
2.Edges: The edges around the tampered car are unnaturally sharp and show signs of
3.being cut out and pasted into the scene.Resolution: The tampered car has a different level of sharpness compared to the rest
of the image, suggesting it was inserted from a different source with a different resolution.
4.Perspective relationship: The size and angle of the tampered car do not fully conform
5.to the perspective of the track and the leading car, making it appear out of place.Shadows: The shadows cast by the tampered car do not match the direction and
quality of the shadows seen elsewhere in the image, indicating that it was not originally part of the scene.
6.Physical laws: The position and angle of the tampered car in relation to the track seem
unnatural, as if it is not properly aligned with the road surface, which defies the physical laws of a real-world racing scenario.
```
```
Fake, √
IoU=0.97
```
```
Fake, √
IoU=0.98
```
Figure 13:More dialogue examples of FakeShield when receiving PhotoShop tampered images.
The first image comes from the Columbia (Ng et al., 2009) dataset and has been tampered with by
Splicing. The elliptical area in the center of the screen is the post-processing area. The second
picture comes from the IMD2020 (Novozamsky et al., 2020) dataset and has been tampered with by
Copy-Move. The car on the right side of the image is copied and pasted from the first car.


```
FakeShield
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
User
```
```
1.1. Whether the picture has been tampered with / Description of the tampered area: 1.The picture has not been tampered with. It is a picture taken directly by the camera.
2.Judgment basis:
1.The lighting in the image appears consistent, with the direction of the sunlight casting shadows on the ground that align witdirection of the light on the horses. h the
2.The colors are natural and there are no signs of unnatural marks or areas that would suggest tampering.
3.The edges of the horses and the surrounding grass are smooth and continuous, with no signs of pixel distortion or unnatural sharpness that would indicate compositing.
4.The resolution across the image is uniform, without any areas of blurring or compression that would suggest editing.
5.The perspective and scale of the horses in relation to the surrounding environment are realistic, and the depth of field is consistent with what would be expected from a camera's lens.
6.The shadows cast by the horses are appropriate for the light source and the physical laws of reflection and shadow casting are
7.observed. There is no text or logos present to assess for tampering. Overall, the image adheres to the real laws of physics and shows no signs of digital manipulation.
```
```
FakeShield
```
```
Please determine whether this picture was taken
directly by the camera, state your conclusion with
reasons, and provide the mask if there is any
tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area: 1. The picture has not been tampered with. It is a picture taken directly by the camera.
2.Judgment basis:
```
1. The lighting in the image appears consistent, with the light source seemingly coming from the front, slightly above the subjeas indicated by the subtle shadows under the chins and instruments. cts,
2.The colors are coherent throughout the scene, with no unusual or out-of-place coloration that might suggest tampering.
3.The edges of the subjects and objects are smooth and continuous, with no signs of unnatural pixel distribution or jagged edgethat would indicate compositing or manipulation. s
4.The resolution across the image is uniform, without any areas of blurring or compression that would suggest editing.
5.The perspective and scale relationships are realistic; the musicians and their instruments are proportionate to each other, athere's a natural depth of field with the background slightly out of focus compared to the foreground, consistent with the nd
    optics of a camera lens.
6.The reflections and shadows in the photograph are appropriate for the visible light sources, and there are no unnatural shadows or reflections that would imply tampering.
7.There is no text or logos present to assess for consistency, but the physical laws appear to be upheld, with no improbable
8.trajectories or reflections. Overall, the image presents a cohesive scene that adheres to the real laws of physics and photography, making it
    distinguishable as a real, unaltered photograph when compared to a collection of tampered images.

```
Real, √
```
```
Real, √
```
Figure 14:Dialogue examples of FakeShield when receiving authentic scene images.The first
image comes from the CASIA1+ (Dong et al., 2013) dataset. The second image comes from the
IMD2020 (Novozamsky et al., 2020) dataset.


```
FakeShield
```
```
Was this picture taken directly by the camera? State
your conclusion and give reasons. If it is tampered with,
please output the mask of the tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, specifically in the lower facial region, including
the mouth and chin area.
2.Judgment basis:
1.The tampered area shows signs of distortion and unnatural blending with the
surrounding skin texture.
2.The lighting on the tampered area does not match the rest of the face, indicating
manipulation.
3.The edges of the mouth and chin appear to be artificially smooth and lack the
natural lines and shadows one would expect to see.
4.The skin tone within the tampered area differs from the adjacent areas, suggesting
inconsistency in skin texture.
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly
by the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, specifically in the region of the glasses on
the left side of the face. The tampered area includes the temple of the glasses,
part of the frame, and a small section of the lens.
2.Judgment basis:
1.The temple of the glasses shows an unnatural straight line that does not conform
to the typical curvature of eyeglass frames.
2.The frame of the glasses has an irregular shape, lacking the continuity expected
from a real frame.
3.The lens of the glasses has a distorted section, which is inconsistent with the
rest of the lens's appearance.
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly
by the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, specifically in the region of the hair on the
right side of the image.
2.Judgment basis:
1.The hair on the right side shows an unnatural transition in color and texture
compared to the left side, indicating a possible alteration.
2.There is a lack of consistency in the hair strands' direction and flow, which is not
typical of how hair naturally falls or behaves.
3.The edge of the tampered area near the hair appears to be unnaturally straight,
which is atypical for the natural irregularity of hair outlines.
```
```
Fake, √
IoU=0.82
```
```
Fake, √
IoU=0.91
```
```
Fake, √
IoU=0.79
```
Figure 15:Dialogue examples of FakeShield when receiving DeepFake tampered images.The
three images all come from the DFFD (Dang et al., 2020) dataset and have been tampered with
by FaceAPP (FaceApp Limited, 2017). The first picture has a beard added to the chin, the second
picture has glasses added, and the third picture has some hair added to the right side of the face.


```
FakeShield
```
```
Was this picture taken directly by the camera?
State your conclusion and give reasons. If it is
tampered with, please output the mask of the
tampered area.
```
```
User
```
```
1.Whether the image has been tampered with/description of the tampered area:
1.The image appears to be authentic and has not been tampered with. The visible parts of the image show natural characteristicsconsistent with a photograph taken directly by a camera.
2.Judgment basis:
1.The visible portion of the image shows a natural skin tone with subtle variations and shadows that are consistent with the lisuggesting that the image is real. ghting,
2.The clothing visible in the image has a realistic texture and the fabric folds align with the natural laws of physics.
3.If this image were placed among deepfake tampered pictures, it would stand out due to the consistency in lighting, natural sktexture, and the absence of any distorted or blurred edges that are common in deepfake images. in
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly by
the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the image has been tampered with/description of the tampered area: 1.The image has not been tampered with. It is a picture of a face taken directly by the camera. The syntax is unchanged and the
2.Judgment basis: semantics are consistent.
1.The visible hair strands have natural variations in color and thickness, which are typical of a real image.
2.3.The lighting on the hair appears consistent, with highlights and shadows that correspond to a natural environment. The texture and color of the skin near the hairline seem authentic and match the expected appearance of a real person.
4.If this image were placed among deepfake tampered pictures, it would stand out due to the natural inconsistencies and
imperfections that are characteristic of a genuine photograph, such as the variation in hair texture and the realistic interaof light with the skin and hair. ction
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly by
the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the image has been tampered with/description of the tampered area:
1.The image appears authentic and has not been tampered with. The visible portion of the image shows natural characteristics consistent with a photograph taken directly by a camera.
2.Judgment basis: 1.The visible part of the image shows a portion of eyeglasses with a realistic reflection and transparency, indicating a high
likelihood of being a genuine photograph.
2.The texture and color of the visible hair seem natural, with individual strands and natural shading that would be difficult treplicate with deepfake technology without noticeable artifacts. o
3.The edge of the eyeglasses against the skin and hair appears consistent with the expected lighting and perspective, without any
4.signs of blurring or distortion that might suggest digital manipulation. If this image were placed among deepfake tampered images, the natural light reflection on the glasses, the hair texture, and the
absence of any unnatural symmetry or disproportionate features would be key indicators to identify it as an authentic image.
```
```
Real, √
```
```
Real, √
```
```
Real, √
```
Figure 16:Dialogue examples of FakeShield when receiving authentic face images.The three
images all come from the DFFD (Dang et al., 2020) dataset.


```
FakeShield
```
```
Was this picture taken directly by the camera?
State your conclusion and give reasons. If it is
tampered with, please output the mask of the
tampered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area: 1.The picture has been tampered with, specifically in the upper left corner of the image,
where the window is located. The tampered area includes the window and the objects visible through it.
2.Judgment basis: The tampering is evident due to several factors:
1.Lighting: The lighting on the objects visible through the window does not match the rest of the scene. They appear to be lit inconsistently with the natural light coming into the
room.
2.Edges: The edges around the window and the objects outside are unnaturally sharp and do not blend seamlessly with the surrounding environment, indicating possible digital insertion.
3.Perspective relationship: The perspective of the objects outside the window seems
inconsistent with the room's interior. The arrangement and proportions of these objects do not align with the expected view through a window.
4.Shadows: There is a lack of natural reflections or shadows that one would expect from objects seen through a window in this setting. This inconsistency suggests that the scene
outside has been artificially altered.
5.Physical laws: The appearance and placement of the objects outside the window seem unrealistic, indicating that they have been digitally added rather than being part of the
original photo.
```
```
FakeShield
```
```
Can you confirm if this picture was taken directly by
the camera? Provide your conclusion along with
reasons. If tampered, please show the mask of the
altered area.
```
```
User
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with. The tampered area is located in the center of the image, covering the entire circular object that appears to be a clock mounted on the wall.
2.Judgment basis: The tampering is evident due to several factors:1.Lighting: The lighting on the clock does not match the surrounding environment. The clock
appears to have a different light source, as it has highlights and shadows that are
2.inconsistent with the rest of the image.Edges: The edges of the clock are unnaturally smooth and lack the texture continuity with
the wall it is attached to.
3.Resolution: The clock's resolution seems to be different from the rest of the image, with the clock appearing slightly blurrier than the surrounding wall texture.
4.Perspective relationship: The clock's perspective does not fully align with the wall it is on. The angle and position of the clock seem unnatural, as if it is not properly affixed to the
wall.
5.Shadows: The shadowing on the clock does not correspond with the light source and the shadows present in the rest of the image, suggesting that the clock was not originally part
of the scene.
6.Physical laws: The way the clock is mounted on the wall appears to defy physical laws, as there is no clear means of support or attachment visible.
```
```
Fake, √
IoU=0.97
```
```
Fake, √
IoU=0.99
```
Figure 17:Dialogue examples of FakeShield when receiving AIGC-Editing tampered images.
The three images all come from the self-generated AIGC-Editing dataset and have been tampered
with by Stable-Diffusion-Inpainting (Lugmayr et al., 2022). The window in the upper left corner of
the first picture has been redrawn, and the circular clock area on the left side of the second picture
has been redrawn.


```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area: The picture has been
tampered with, in the upper section, spanning across the entire width of the image. The tampered area includes
the mountain range and the sky immediately above it.
2.Judgment basis:
1.The tampering is evident by the unnaturally smooth transition between the mountain tops and the sky,
lacking detailed texture that would be expected in a natural setting.
2.There is a lack of consistent lighting and shadows on the mountains which do not match with the rest of the
image, indicating that this part of the image may have been inserted or altered. The edges of the mountains
against the sky are unusually crisp without gradation or atmospheric haze that would normally occur due to
3.distance. Additionally, the pixelation around the edges of the mountains does not match the resolution of the
foreground, suggesting a composite of two different images.
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area: The picture has been
tampered with, specifically in the area around the lighthouse. The tampered area is located in the upper right
corner of the image, against the horizon where the sky meets the sea.
2.Judgment basis: There are several details that suggest tampering:1.Lighting: The lighthouse appears to be unnaturally bright compared to the surrounding environment,
especially given that the sun is setting and the light is diffused.
2.Edges: The lighthouse has a somewhat blurry outline that does not match the sharpness of the horizon line,
indicating it may have been artificially inserted or altered.
3.Shadows: The lighthouse does not cast a shadow that is consistent with the setting sun's direction and the
existing shadows in the image.
4.Perspective relationship: The scale of the lighthouse in relation to its placement on the horizon seems
slightly off, making it appear as if it does not quite belong there.
5.Physical laws: The lighthouse's reflection in the water does not match the angle of the sunlight or the other reflections in the water, breaking the consistency expected in a real photo.
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the picture has been tampered with / Description of the tampered area: The picture has not been tampered with. It is a picture taken directly by
    the camera.
2. Judgment basis:
    1.Lighting: The lighting in the image appears natural and consistent. The light source seems to be coming from outside the window,casting shadows and
       highlights appropriately on the window sill and the objects placed on it. The soft diffuse light suggests an overcast day or a shaded area outside,
       common in real-life scenarios.
    2.Copy and Paste: There is no visible sign of duplication or mirroring that would suggest a copy and paste operation. Each object has unique reflections,
       colors, and shadows.
    3.Edges: The edges of the objects, window, and foliage outside are smooth and continuous, with no visible signs of pixel distorwould indicate tampering. tion or abrupt changes that
    4.Resolution: The resolution of the image appears uniform, with no areas showing signs of blurring or jaggedness that would suggest a composite image.
    5.Perspective: The perspective and scale of the objects on the window sill are consistent with how they would appear in real life.The size of the objects
       relative to each other and their placement gives a sense of depth that aligns with normal visual perception.

```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the picture has been tampered with / Description of the tampered area: The image appears not to have been tampered wipicture taken directly by the camera. th. It seems to be a
2. Judgment basis:
    1.Lighting: The lighting on the climber is consistent with the direction of sunlight, casting a natural shadow on the climber's body, which conforms to the
       direction and quality of the lighting in the environment.
    2.Copy and Paste: There are no signs of duplication or mirroring that would indicate a copy and paste operation. \n- Edges: The edges of the climber and
       the rock face appear natural with no signs of unnatural pixel distribution or obvious traces of clipping.
    3.Resolution: The image maintains a consistent resolution throughout, with no unnatural blurring or jaggedness that would suggest tampering.
    4.Perspective: The scale and perspective of the climber against the rock face are realistic, and the proportions seem accurate.
    5.Shadows: The shadows cast by the climber are consistent with the light source coming from the upper left, and there are no unnatural shadows that
    6.would imply manipulation.Text: There is no visible text or logos to assess for consistency.
    7.Physical laws: The climber's position and equipment behave as expected under the force of gravity, and there are no violations of physical laws.

Figure 18: MMTD-Set data samples related to PhotoShop tampering. The four images are all from
the CASIAv2 data set. The first two pictures have been tampered with, but the last two pictures have
not been tampered with.


```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, the tampered area is on the right side of the image, extending vertically from the top to the bottom of the frame. It includes the edge of a bathroom wall and a portion of a black pedestal fan that
stands next to the sink.
2.Judgment basis: The tampered area shows several inconsistencies. 1.The edge of the wall on the right side has a jagged, unnatural outline that does not align with the typical straight lines
found in interior architecture.
2.Moreover, the pedestal fan has a distorted shape, with the top part of the fan appearing disconnected and floating without support.
3.The shadows and lighting in the tampered area do not match the rest of the image; the shadow on the tampered part of
the fan is inconsistent with the lighting on the rest of the fan and the surrounding environment.
4.There are no clear reflections or shadows corresponding to the tampered section of the wall, suggesting an unnatural alteration. The resolution in the tampered area is also inconsistent, with some parts appearing blurrier than the rest of
the image.
5.These signs indicate that the image has been edited."
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area: The picture has been tampered with. The tampering is located on the screen of the Nokia phone, occupying almost the entire screen area, with only a small
untampered margin visible around the edges.
2.Judgment basis: 1.The contents of the tampered area include jumbled and inverted text, numbers, and symbols that do not conform to
the typical display of a phone screen.
2.The colors within the tampered area are inconsistent with the original screen display, showing a mix of orange, yellow, and white that appear unnatural.
3.The text and numbers are also not aligned or positioned as expected for a phone's user interface, suggesting a
manipulation of the original content. Additionally, there is a lack of continuity with the phone's expected graphical interface, which would normally display information in an organized and legible manner."
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the picture has been tampered with / Description of the tampered area: The picture has not been tampered with. It appears to be a genuine photograph
    captured directly by a camera.
2.Judgment basis: 1.The lighting in the image is consistent, with the shadows on the cake and the hands indicating a light source coming from the upper left side, which aligns with
    the natural shadow casting.
2.There are no obvious signs of duplication or mirroring, and the edges of the subjects and objects are natural without any disdistributions. continuities or irregular pixel
3.The resolution and quality of the image are uniform, with no areas showing signs of blurring or excessive compression that would suggest manipulation. The
perspective and scale of the objects and people are realistic and proportionate, and the depth of field is consistent, with aback of the image. natural focus drop-off towards the
4.The reflections and shadows present in the image behave as expected under the lighting conditions. Text and logos visible in the background are consistent
with their surroundings. All content in the picture adheres to physical laws without any anomalies."

```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the picture has been tampered with / Description of the tampered area: The image appears not to have been tampered with. It looks like a photograph
2.taken directly by a camera.Judgment basis: The following details support the conclusion that this is a real, untampered photograph:
    1.Lighting: The lighting on the salad and the glass of liquid appears consistent, with soft, diffused light coming from one general direction, which is plausible for
    2.an indoor setting. There are no harsh shadows or inconsistent light spots that would suggest tampering.Copy and Paste: There are no obvious repeating patterns or mirrored sections that would indicate a copy and paste operation has been conducted.
    3.Edges: The edges of the salad leaves, fork, and other elements in the image look natural with no visible signs of harsh or artificial edges that would suggest
       cutting or compositing.would indicate tampering. \n\n- Resolution: The image resolution seems consistent throughout the photograph, with no areas of unnatural blurring or pixelation that
    4.Perspective: The perspective and scale of the objects, such as the fork, salad, and glass, are consistent with what would be expected in a real-life setting.
    5.Shadows: The shadows cast by the fork and the leaves in the salad are subtle and match the lighting of the scene. There are nsuggest manipulation. o unnatural shadows that would

Figure 19: MMTD-Set data samples related to AIGC-Editing tampering. The four images are all
from the self-generated AIGC-Editing dataset. The first two pictures have been tampered with, but
the last two pictures have not been tampered with.


```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area: The picture has been tampered with,
specifically the hair and ears. The tampered areas include the top and sides of the head, extending to cover the full hair region
and the ears on both sides.
2.Judgment basis: 1.The hair appears unnatural and lacks consistent texture when compared to typical hair, which would normally have more
variation in color and strand definition.
2.The edges of the hair near the forehead and temples are too sharp and show no natural hairline; real hair usually has more
3.irregularity at the hairline.There is a lack of depth and natural shading around the ears, which can be expected if the image was manipulated; ears
typically cast subtle shadows and have more defined contours.
4.The overall hair shape and the way it sits on the head do not conform to natural hair physics, indicating digital alteration."
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
```
1.Whether the picture has been tampered with / Description of the tampered area:
1.The picture has been tampered with, specifically in the lower facial region including the jawline and the chin, as well as a
2.Judgment basis:small portion of the neck.
1.The tampered area appears to have an inconsistent skin tone compared to the surrounding untampered skin, indicating a
possible mismatch in texture and color.
2.The edges of the tampered area around the jawline and chin show signs of being unnaturally smooth, lacking the typical definition and texture found in these facial features.
3.There is a lack of natural shadowing and lighting in the tampered area, which does not correspond with the lighting
present in the rest of the image."
```
```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the image has been tampered with/description of the tampered area: The image does not appear to have been tampered wiimage, including hair and ears, display natural variations and asymmetries typical of a real photograph taken directly by a camera. th. The visible parts of the
2. Judgment basis: The hair shows natural variations in texture and color, with individual strands and different shades that are typical of real hair. The ear is
    partially visible and displays a natural contour and complexion consistent with the visible hair, suggesting that the image is of a real person. To recognize it from a
    bunch of deepfake tampered pictures, one would look for inconsistencies in skin texture, unnatural symmetry, or artifacts aroare present in this image. The natural lighting and lack of any visible digital artifacts also support the conclusion that the image is authentic and has not been und the hair or ears—none of which
    manipulated."

```
Was this image captured straight from the camera?
Share your conclusion and reasons. If modifications
are detected, output the tampered area's mask.
```
1. Whether the image has been tampered with/description of the tampered area: The image appears to be authentic and has not been tampered with. Visible parts of
2.the image show natural features and consistent lighting.Judgment basis: The visible hair shows natural color variations and texture consistent with real hair. The lighting on the hair suggests a single light source, which
    is coherent and creates realistic shadows and highlights. The portion of the ear visible in the image has natural contours and skin tone. The visible clothing appears
    to have a consistent texture and sits on the body in a way that aligns with the natural folds expected from the fabric and the pull of gravity. These details
    indicate that the image is likely a real photograph and would stand out as authentic among deepfake tampered pictures due to coherent lighting that deepfake images often lack." the natural inconsistencies and the

Figure 20: MMTD-Set data samples related to DeepFake tampering. The four images are all from
the DFFD (Dang et al., 2020) dataset. The first two pictures have been tampered with, but the last
two pictures have not been tampered with.