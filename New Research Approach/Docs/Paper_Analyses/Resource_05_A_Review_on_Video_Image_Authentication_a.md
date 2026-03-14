# Resource 05: A_Review_on_Video_Image_Authentication_a.pdf

## 1. Resource Overview
- Title: A Review on Video/Image Authentication and Temper Detection Techniques
- Source: Research paper PDF
- Author: Author not identified from local resource
- Year: 2013

This is an older authentication review. Its objective is to organize video and image authentication methods, including both active and passive approaches, rather than to solve modern tamper localization with deep learning.

## 2. Technical Summary
The local project inventory describes it as a taxonomy-style review covering digital signatures, watermarking, hashing, blind deconvolution, and related authentication techniques. That means it is mostly about categories of evidence and security paradigms, not about a concrete learning system with mask prediction.

Technically, the paper helps explain how active and passive forensics differ. Beyond that, it is too old and too broad to drive modern architectural decisions.

## 3. Key Techniques Used
- Active authentication methods such as signatures and watermarking
- Passive authenticity analysis
- High-level taxonomy of media verification strategies

These techniques are historically important because they show tamper detection is not one problem. Some methods assume capture-time control, while passive forensics assumes you get an already-finished image and must infer manipulation after the fact.

## 4. Strengths of the Approach
Its strength is framing. If the project needs a short explanation of active versus passive forensics, this paper can provide that context.

It is also useful as a reminder that many localization claims in active-authentication papers are solving a fundamentally different task.

## 5. Weaknesses or Limitations
This paper is dated. It predates the modern deep-learning era that actually matters for the assignment. Using it to justify current model design would make the literature review look stale fast.

It also does not directly support pixel-wise localization on natural-image tampering datasets such as CASIA. So if it starts influencing the architecture section, the project has already gone off the rails.

## 6. Alignment With Assignment
Alignment: Low

The assignment needs detection, localization, and practical architecture reasoning on Colab-class hardware. This resource mostly supplies historical taxonomy, not a workable design path.

## 7. Relevance to My Project
Useful parts:
- One paragraph of background on active versus passive forensics

Unnecessary parts:
- Any attempt to use it as evidence for modern model selection
- Watermarking, signing, and other controlled-capture methods

## 8. Should This Be Used?
Do not use.

At most, keep it as background reading. It should not influence the actual design, evaluation, or implementation decisions for this assignment.

## 9. Integration Ideas
- If a literature review section needs a short taxonomy paragraph, cite the active-versus-passive distinction and move on.
- Do not let it shape the model architecture or benchmark plan.

## 10. Citation
A Review on Video/Image Authentication and Temper Detection Techniques. Local PDF copy: `Research Papers/A_Review_on_Video_Image_Authentication_a.pdf`. Year identified from local project notes as 2013. Author not identified from local resource.
