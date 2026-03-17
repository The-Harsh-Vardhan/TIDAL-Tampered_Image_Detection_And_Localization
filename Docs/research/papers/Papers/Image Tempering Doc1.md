# **Advanced Methodologies in Digital Image Forensics and the Structural Dynamics of Scientific Knowledge Accessibility**

The contemporary digital landscape is defined by an unprecedented proliferation of visual information, where digital images serve as fundamental pillars of evidence in journalism, jurisprudence, and social discourse.1 However, the concurrent advancement of accessible, high-performance image manipulation software has compromised the inherent trust traditionally placed in photographic evidence.3 Digital image forensics has consequently emerged as a critical field of computational research, dedicated to developing robust algorithms capable of identifying subtle traces of tampering that escape human perception.5 This report provides an exhaustive analysis of five seminal research papers published via the Institute of Electrical and Electronics Engineers (IEEE), examining their technical contributions to forgery detection and the broader socio-technical mechanisms governing their accessibility and dissemination in the global research community.7

## **Taxonomic Evolution of Image Tampering Detection**

The field of image forensics is broadly categorized into active and passive (blind) techniques.6 Active methods, such as digital watermarking and steganography, require the embedding of a signature at the time of image creation.11 Passive methods, which form the core of the research analyzed herein, rely solely on the analysis of intrinsic statistical artifacts left by the imaging pipeline or the manipulation process itself.5 The transition from traditional hand-crafted feature extraction to deep-learning-based end-to-end models represents the most significant shift in the recent history of this domain.4

### **Computational Paradigms in Forgery Localization**

The identification of image forgeries involves two distinct but related tasks: classification—determining if an image has been altered—and localization—identifying the specific spatial coordinates of the modification.2 Traditional approaches often struggle with "wild" images that have undergone multiple post-processing steps such as JPEG re-compression, blurring, or noise addition, which are frequently employed by forgers to mask tampering boundaries.4 Modern architectures, particularly those leveraging semantic segmentation frameworks like U-Net, address these challenges by treating forgery localization as a pixel-wise labeling problem.14

## **Technical Analysis of ELA-CNN Hybrid Frameworks**

The research presented in "Tempered Image Detection Using ELA and Convolutional Neural Networks" (IEEE Document 10444440\) addresses the urgency of identifying falsified visuals in the era of fake news.1 This methodology centers on the synergy between Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs).16

### **Mechanism of Error Level Analysis**

Error Level Analysis operates on the premise that digital images, specifically those in the JPEG format, lose a predictable amount of information each time they are compressed.16 In an authentic image, the compression error level is relatively uniform across the entire frame. However, when a region is modified—for instance, through splicing a portion of another image—and subsequently resaved, the modified area will exhibit a distinct compression signature compared to the original background.4

By calculating the difference between an image and its version resaved at a known quality level (e.g., 95%), ELA highlights these discrepancies as variations in color shades.16 This pre-processing step is critical because it amplifies high-frequency artifacts that are otherwise invisible in the standard RGB color space, providing a more informative input for neural architectures.4

### **Architectural Implementation and Efficiency**

The model proposed by Mishra et al. utilizes a streamlined CNN architecture designed to process these ELA-enhanced images.16 The network configuration involves:

* **Convolutional Layers:** Two ![][image1] convolutional layers responsible for extracting spatial features such as edges and texture discrepancies within the ELA map.16  
* **Pooling and Regularization:** A MaxPooling layer to reduce spatial dimensions and a Dropout layer to prevent overfitting by randomly deactivating neurons during training.16  
* **Fully Connected Layers:** A dense layer followed by a Softmax output layer for final classification.16

The integration of ELA significantly accelerates model convergence. The research demonstrates that the network achieves an accuracy of 87.75% within just 10 epochs.16 This efficiency is a direct result of the ELA pre-processing, which reduces the computational burden on the convolutional filters by highlighting relevant forensic features prior to the learning phase.16

| Parameter | Configuration | Outcome |
| :---- | :---- | :---- |
| Pre-processing | Error Level Analysis (ELA) | Enhanced visual cues for compression |
| Network Backbone | 2D CNN | 87.75% Accuracy 16 |
| Convergence Rate | 10 Epochs | Reduced training cost |
| Dataset | CASIA2 \+ Kaggle Merged | Robust generalization 16 |

## **Integrated Classification and Segmentation for Realistic Tampering**

The document "Real or Fake? A Practical Method for Detecting Tempered Images" (IEEE Document 10052973\) moves beyond simple classification to propose a dual-task end-to-end model.2 This research emphasizes the challenges posed by accessible editing tools like Photoshop, which allow for seamless splicing, inserting, and removing of objects.2

### **End-to-End Deep Learning Architecture**

Traditional forensic workflows often employ separate models for detecting if an image is fake and for locating the forged parts. The method described in Document 10052973 merges these tasks into a single framework, arguing that the shared feature representation in the network's backbone can enhance performance in both classification and segmentation.2 This approach utilizes a deep learning backbone to automatically extract high-level semantic features, which are then branched into:

1. **Classification Branch:** Outputs a binary result (Real or Fake) based on the global features of the image.2  
2. **Segmentation Branch:** Produces a pixel-level map indicating the forged regions.2

### **Forensic Implications of Double Quantization**

A key forensic indicator discussed in this context is the Double Quantization (DQ) effect.2 When a JPEG image is tampered with and resaved, the authentic regions typically undergo a second round of quantization, whereas the tampered regions may only experience one (if they were sourced from an uncompressed or differently compressed original).2 Analyzing the DQ values allows the model to differentiate between segments that have a consistent compression history and those that do not, a technique that proves effective across multiple image formats beyond just JPEG.2

## **Geometric Invariance and Optimization in Copy-Rotate-Move Detection**

Copy-move forgery, where a region of an image is duplicated and moved to another location within the same frame, is particularly difficult to detect when the copied region is rotated or scaled.18 The research "Robust and Optimized Algorithm for Detection of Copy-Rotate-Move Tempering" (IEEE Document 10168896\) provides a computationally efficient solution to this problem.8

### **Zernike Moments for Rotation Invariance**

The core of this algorithm is the use of Zernike moments, which are orthogonal moments defined over the unit disk.20 Unlike standard pixel-matching or Fourier-based methods, Zernike moments possess a unique property of rotation invariance—the magnitude of the moment remains constant regardless of the orientation of the underlying image patch.20

The Zernike moment of order ![][image2] and repetition ![][image3] is calculated as:

![][image4]  
where ![][image5] and ![][image6] are the Zernike polynomials.20 By extracting these features from overlapping image blocks, the algorithm can identify duplicated regions even if they have been subjected to significant geometric transformations.18

### **Metaheuristic Optimization via Ant Colony Algorithms**

To address the high computational complexity of matching features across large images, the authors integrate Ant Colony Optimization (ACO).20 ACO is an evolutionary metaheuristic that mimics the pheromone-trail-following behavior of ants to find optimal paths.20 In the context of copy-move detection, the "ants" explore the feature space to identify the most likely matches, significantly reducing the search time compared to exhaustive lexicographical sorting.18

| Feature Extraction | Optimization | Accuracy Rate |
| :---- | :---- | :---- |
| Zernike Moments | Ant Colony Optimization | 98.44% 18 |
| SIFT \+ SVD | Hybrid Matching | Robust to scaling 18 |
| DCT Coefficients | Lexicographical Sorting | Susceptible to rotation 15 |

## **Semantic Segmentation and Mixed Tampering Localization via U-Net**

The paper "Deep Localization on Mixed Image Tempering Techniques Using U-Net" (IEEE Document 10652417\) focuses on the "mixed tampering" scenario, where an image may contain multiple types of forgeries (e.g., both splicing and copy-move) simultaneously.4

### **U-Net Architecture and Feature Fusion**

The U-Net architecture is favored for this task due to its symmetrical encoder-decoder structure.14 The encoder captures the contextual information of the image through successive convolutional and pooling layers, while the decoder uses transposed convolutions to restore the spatial resolution, enabling the model to assign a "tampered" or "authentic" label to every individual pixel.14

The research by Srivastava et al. introduces a "forgery-aware guided spatial-frequency feature fusion network".4 This model does not rely solely on RGB data; instead, it integrates features from the frequency domain, allowing it to detect inconsistencies in the underlying signal structure that are caused by disparate image origins.4 This fusion-enhanced network (FENet) utilizes adaptive feature fusion and edge attention mechanisms to maintain high boundary accuracy even when the image quality has been degraded by post-processing.4

### **Performance on Standard Benchmarks**

Evaluation using the CASIA\_v2 dataset, which contains 7,541 authentic and 5,124 tampered images, indicates that this U-Net-based approach achieves a classification accuracy of approximately 95%.4 The model is particularly effective at generating interpretable visual feedback via predicted forgery masks, though it notes that copy-move forgeries remain slightly more challenging to localize than splicing due to their inherent self-similarity within the image.4

## **Cryptographic Integrity and Hybrid Computer Vision Frameworks**

The research "Image Forgery Detection Using MD5 & Open CV" (IEEE Document 10895348\) proposes a dual-layered security approach that combines cryptographic verification with forensic content analysis.7

### **MD5 Hashing for Integrity Verification**

The Message Digest Algorithm 5 (MD5) is employed to generate a unique digital fingerprint for an original image.10 This 128-bit hash value serves as a baseline for integrity verification. Any unauthorized modification to the file, no matter how minor, will result in a completely different hash value, a property known as the avalanche effect.22 The hash ![][image7] of image ![][image8] is defined such that:

![][image9]  
If ![][image8] is tampered to create ![][image10], then ![][image11] with near-certainty.22 However, because MD5 is sensitive to non-malicious changes (like re-saving or metadata edits), it is used primarily as a fast initial screening tool to flag potential alterations.10

### **OpenCV for Deep Anomaly Detection**

To provide a more granular analysis, the system integrates the OpenCV library to perform advanced feature extraction.10 This layer analyzes:

* **Texture and Color Profiling:** Identifying inconsistencies in the local distribution of pixels that might indicate a spliced object.22  
* **Shape Recognition:** Detecting anomalies in object boundaries that often occur during manual image editing.22  
* **Pixel Pattern Analysis:** Using OpenCV's computational tools to discover subtle irregularities that simpler detection methods might overlook.10

The resulting hybrid framework, dubbed "UEFRG," achieves a reduction in false positives by ensuring that only modifications impacting the visual or structural content of the image are flagged as forgeries, rather than benign file system changes.10

## **Socio-Technical Landscape of Research Accessibility**

A primary objective of the initial inquiry was the acquisition of these IEEE documents for free. The accessibility of these specific papers varies significantly based on their Open Access status, author self-archiving practices, and the policies of the IEEE Xplore digital library.7

### **Open Access vs. Subscription Models**

Academic publishing is currently transitioning toward Open Access (OA) models, which allow the public to read research without a subscription.8 Within the requested documents, "Robust and Optimized Algorithm for Detection of Copy-Rotate-Move Tempering" (10168896) is officially designated as **Open Access** and is available under a Creative Commons License.8 This permits any user to download the full PDF directly from the IEEE Xplore portal.8

Conversely, other documents such as 10444440, 10052973, and 10652417 are typically restricted behind institutional paywalls or require individual purchase.7 These documents are associated with "Purchase Details" and "Order History" on the publisher's site, signifying that they are not part of the standard OA program.9

### **Role of ResearchGate and Preprints in Dissemination**

ResearchGate has emerged as a vital alternative for accessing paywalled research.10 Authors often exercise their right to share "preprints" or "accepted manuscripts" on their personal profiles.10

* **Document 10895348:** The lead author, Mohammad Shahnawaz Shaikh, uploaded the full-text PDF of "Image Forgery Detection Using MD5 & Open CV" to ResearchGate on March 04, 2025\.10 This allows users to legally access the complete conference paper for free.10  
* **Document 10652417:** While the work is listed on ResearchGate, the author (Ankit Kumar Srivastava) has not yet claimed the profile or added the full text, meaning only metadata is currently available on that platform.24

| Paper ID | Primary Access Path | Status |
| :---- | :---- | :---- |
| 10444440 | IEEE Subscription / Excerpts in Snippets | Paywalled 16 |
| 10052973 | IEEE Subscription / Excerpts in Snippets | Paywalled 2 |
| 10168896 | IEEE Xplore Direct Download | **Open Access** 8 |
| 10652417 | IEEE Subscription / ResearchGate Request | Paywalled 24 |
| 10895348 | ResearchGate Full-Text | **Free (Author Upload)** 10 |

## **Comparative Metrics and Experimental Validation**

The effectiveness of the proposed forensic algorithms is validated through rigorous testing on standardized datasets. Accuracy and the F1-score are the primary metrics used to assess performance.4

### **Benchmarking against State-of-the-Art (SOTA)**

The research indicates that deep learning models consistently outperform traditional block-based or pixel-based methods.4 For example, the transfer learning approach using EfficientNetV2B0 achieves superior generalization compared to older models like MVSS-Net++ or DRRU-Net.13 Similarly, the use of a modified U-Net with regularization achieved an F1-score of 0.96 on test sets, illustrating the power of deep learning for precise localization.14

| Methodology | Dataset | Performance Metric |
| :---- | :---- | :---- |
| ELA \+ CNN | CASIA2 | 87.75% Accuracy 16 |
| Regularized U-Net | Validation Set | 0.96 F1 Score 14 |
| MobileNetV2 | Splicing/Copy-Move | 95% Accuracy 14 |
| Zernike \+ ACO | MICC-F220 | 98.44% Accuracy 18 |
| LBP \+ CNN | CASIA-2.0 | High efficiency 4 |

### **Handling Post-Processing Attacks**

A robust forensic model must maintain high detection rates even when images are subjected to "attacks" intended to destroy forensic evidence.15 These include:

* **JPEG Compression:** Many algorithms see a significant drop in accuracy as the quality factor (QF) decreases.6 However, methods leveraging ELA are specifically designed to exploit these compression artifacts.4  
* **Gaussian Filtering and Blurring:** Smoothing the edges of a spliced region is a common tactic. The blur estimation method presented in earlier forensic literature can detect these manually applied low-pass filters.15  
* **Additive Noise:** Forgers may add Gaussian noise to an image to homogenize the noise floor. Advanced models using Noise Pattern Analysis (NPA) can still identify inconsistencies in the sensor noise patterns of the spliced region versus the background.5

## **Conclusion and Future Research Trajectories**

The analysis of these five IEEE papers reveals a field in rapid transition.5 The move toward deep learning, specifically the use of U-Net for localization and ELA-CNN for classification, has significantly raised the bar for forgery detection accuracy.14 Simultaneously, the use of cryptographic tools like MD5 provides a necessary layer of file-level integrity that complements purely visual analysis.10

For researchers and professionals seeking to access this literature, the availability of Document 10168896 as Open Access and 10895348 via author self-archiving reflects a growing trend toward transparency in the scientific record.8 As image manipulation techniques continue to evolve—driven by the advent of generative AI and deepfakes—the future of image forensics will likely lie in "multi-domain fusion" models that combine spatial, frequency, and metadata analysis into a single, cohesive forensic workflow.5 The insights generated from these papers provide the foundational frameworks necessary to maintain digital trust in an increasingly uncertain visual era.

#### **Works cited**

1. Tempered Image Detection Using ELA and Convolutional Neural Networks \- IEEE Xplore, accessed March 11, 2026, [https://ieeexplore.ieee.org/document/10444440/](https://ieeexplore.ieee.org/document/10444440/)  
2. (PDF) Real or Fake? A Practical Method for Detecting Tempered ..., accessed March 11, 2026, [https://www.researchgate.net/publication/366143989\_Real\_or\_Fake\_A\_Practical\_Method\_for\_Detecting\_Tempered\_Images](https://www.researchgate.net/publication/366143989_Real_or_Fake_A_Practical_Method_for_Detecting_Tempered_Images)  
3. Real or Fake? A Practical Method for Detecting Tempered Images \- IEEE Xplore, accessed March 11, 2026, [https://ieeexplore.ieee.org/iel7/10052787/10052789/10052973.pdf](https://ieeexplore.ieee.org/iel7/10052787/10052789/10052973.pdf)  
4. Deep Learning for Localization of Mixed Image Tampering Techniques \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/362770795\_Deep\_Learning\_for\_Localization\_of\_Mixed\_Image\_Tampering\_Techniques](https://www.researchgate.net/publication/362770795_Deep_Learning_for_Localization_of_Mixed_Image_Tampering_Techniques)  
5. (PDF) Digital image forensics: A booklet for beginners \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/220664038\_Digital\_image\_forensics\_A\_booklet\_for\_beginners](https://www.researchgate.net/publication/220664038_Digital_image_forensics_A_booklet_for_beginners)  
6. Detecting doctored JPEG images via DCT coefficient analysis \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/225180603\_Detecting\_doctored\_JPEG\_images\_via\_DCT\_coefficient\_analysis](https://www.researchgate.net/publication/225180603_Detecting_doctored_JPEG_images_via_DCT_coefficient_analysis)  
7. Image Forgery Detection Using MD5 & Open CV | IEEE Conference ..., accessed March 11, 2026, [https://ieeexplore.ieee.org/document/10895348](https://ieeexplore.ieee.org/document/10895348)  
8. Robust and Optimized Algorithm for Detection of Copy-Rotate-Move ..., accessed March 11, 2026, [https://ieeexplore.ieee.org/abstract/document/10168896](https://ieeexplore.ieee.org/abstract/document/10168896)  
9. Deep Localization on Mixed Image Tempering Techniques Using U ..., accessed March 11, 2026, [https://ieeexplore.ieee.org/abstract/document/10652417](https://ieeexplore.ieee.org/abstract/document/10652417)  
10. (PDF) Image Forgery Detection Using MD5 & Open CV, accessed March 11, 2026, [https://www.researchgate.net/publication/389424875\_Image\_Forgery\_Detection\_Using\_MD5\_Open\_CV](https://www.researchgate.net/publication/389424875_Image_Forgery_Detection_Using_MD5_Open_CV)  
11. Image Forgery Detection A survey \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/224397411\_Image\_Forgery\_Detection\_A\_survey](https://www.researchgate.net/publication/224397411_Image_Forgery_Detection_A_survey)  
12. Digital Image Forgery Detection Using JPEG Features and Local Noise Discrepancies, accessed March 11, 2026, [https://www.researchgate.net/publication/263324028\_Digital\_Image\_Forgery\_Detection\_Using\_JPEG\_Features\_and\_Local\_Noise\_Discrepancies](https://www.researchgate.net/publication/263324028_Digital_Image_Forgery_Detection_Using_JPEG_Features_and_Local_Noise_Discrepancies)  
13. Performance Analysis of ELA-CNN model for Image Forgery Detection \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/372256575\_Performance\_Analysis\_of\_ELA-CNN\_model\_for\_Image\_Forgery\_Detection](https://www.researchgate.net/publication/372256575_Performance_Analysis_of_ELA-CNN_model_for_Image_Forgery_Detection)  
14. Image Forgery Detection & Localization Using Regularized U-Net \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/349189771\_Image\_Forgery\_Detection\_Localization\_Using\_Regularized\_U-Net](https://www.researchgate.net/publication/349189771_Image_Forgery_Detection_Localization_Using_Regularized_U-Net)  
15. Image Forgery Localization via Integrating Tampering Possibility Maps \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/312648622\_Image\_Forgery\_Localization\_via\_Integrating\_Tampering\_Possibility\_Maps](https://www.researchgate.net/publication/312648622_Image_Forgery_Localization_via_Integrating_Tampering_Possibility_Maps)  
16. Tempered Image Detection Using ELA and Convolutional Neural Networks \- IEEE Xplore, accessed March 11, 2026, [https://ieeexplore.ieee.org/iel7/10444098/10444131/10444440.pdf](https://ieeexplore.ieee.org/iel7/10444098/10444131/10444440.pdf)  
17. Real or Fake? A Practical Method for Detecting Tempered Images \- IEEE Xplore, accessed March 11, 2026, [https://ieeexplore.ieee.org/document/10052973/](https://ieeexplore.ieee.org/document/10052973/)  
18. Image Copy-Move Forgery Detection Algorithms Based on Spatial Feature Domain, accessed March 11, 2026, [https://www.researchgate.net/publication/350148279\_Image\_Copy-Move\_Forgery\_Detection\_Algorithms\_Based\_on\_Spatial\_Feature\_Domain](https://www.researchgate.net/publication/350148279_Image_Copy-Move_Forgery_Detection_Algorithms_Based_on_Spatial_Feature_Domain)  
19. Exploring Pseudo-Zernike Moments for Effective Image Forgery Detection \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/382735972\_Exploring\_Pseudo-Zernike\_Moments\_for\_Effective\_Image\_Forgery\_Detection](https://www.researchgate.net/publication/382735972_Exploring_Pseudo-Zernike_Moments_for_Effective_Image_Forgery_Detection)  
20. \[PDF\] Fast Computation of Hahn Polynomials for High Order Moments \- Semantic Scholar, accessed March 11, 2026, [https://www.semanticscholar.org/paper/Fast-Computation-of-Hahn-Polynomials-for-High-Order-Mahmmod-Abdulhussain/13d8c70203155c4713e9b8991a208da4108c3e7c](https://www.semanticscholar.org/paper/Fast-Computation-of-Hahn-Polynomials-for-High-Order-Mahmmod-Abdulhussain/13d8c70203155c4713e9b8991a208da4108c3e7c)  
21. Exposing Digital Forgeries by Detecting Traces of Re-sampling \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/220325432\_Exposing\_Digital\_Forgeries\_by\_Detecting\_Traces\_of\_Re-sampling](https://www.researchgate.net/publication/220325432_Exposing_Digital_Forgeries_by_Detecting_Traces_of_Re-sampling)  
22. Exploring the Role of Artificial Intelligence in Image Forgery Detection and Prevention: A Focus on MD5 and Open CV \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/392067751\_Exploring\_the\_Role\_of\_Artificial\_Intelligence\_in\_Image\_Forgery\_Detection\_and\_Prevention\_A\_Focus\_on\_MD5\_and\_Open\_CV](https://www.researchgate.net/publication/392067751_Exploring_the_Role_of_Artificial_Intelligence_in_Image_Forgery_Detection_and_Prevention_A_Focus_on_MD5_and_Open_CV)  
23. The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation \- arXiv.org, accessed March 11, 2026, [https://arxiv.org/html/2602.10500v2](https://arxiv.org/html/2602.10500v2)  
24. Ankit Kumar Srivastava's research works | Maharaja Institute of ..., accessed March 11, 2026, [https://www.researchgate.net/scientific-contributions/Ankit-Kumar-Srivastava-2291482250](https://www.researchgate.net/scientific-contributions/Ankit-Kumar-Srivastava-2291482250)  
25. (PDF) Detecting digital tampering by blur estimation \- ResearchGate, accessed March 11, 2026, [https://www.researchgate.net/publication/4221690\_Detecting\_digital\_tampering\_by\_blur\_estimation](https://www.researchgate.net/publication/4221690_Detecting_digital_tampering_by_blur_estimation)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAABjElEQVR4Xu2TvyuGURTHj1BESRSizAYZJAlZDK/BxGA3KNkM8gcYDEpSSigzqwWDrBYDkQlJUQaDAfnx/T7n3vc973nfZzSo51Ofep5z7nPvfc65VyTjP1EJO+AIbAvvaUzDXXgfPIfbcDM4LzpHWa7hI3yGP/AVVhSNKNANJ0XHncGp8E6XRb/9gLPxA8LJ1uGgDYI5eAVbXDzSDj/hkE8EcvAbNsRAPTwJ8jnSJzpw1MQsXOASNvtEgG14gD0xwD9ahRuwKgZBr2hpxk3MsgD3pfgbC3t0B8dskIv5fvBP3kQX9HByLjLjE4Yu+CLpG03oF12k0ycCLBv7k1Y2siRakbQeJ5PfwAufMLBsnCStbLHvHOMrlcATcgxPYavLRWLZOEkaA/A9WEK16GXbk8KRZAl5jC18vxUtXTm4kS3RjfCQlbAoulCtie2IHnNL7A+Pdjkm4JdoZfJ3KMIkd+DlrW8045rgUcixTxH2ZBgewjVYZ3J5bOO8B7AmjON98HnrE1wJYzMyMv6AX6rbWFa/5at0AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAtklEQVR4XmNgGAWDDYgBMTMSXwCLGBjwA/EWIPYF4v9AfAaIl0HlhIH4HBBfhfLBwAWIlwNxOgNEwx4GiCEgwAPEB4D4IZQPBp5QvAaI/wGxB5KcEhA/Z0CzAQZ+A/FWIOaA8lkYIDajGwIHIOeUI/EVgfgJEF8HYnEgjgFidpgkyDSQDTYwAQaEn1qh/MlAzAiTNAbi+cgCQKAPxK+BeD8QX0QSBwNQWLOiCzJA/CMJpUcB7QAAc/QdgqUT/vMAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAABE0lEQVR4Xu2SsUoDQRRFXzCFECFBxBAQgnapLMRC0GBpCpu0/oJVGts0/kCaQHqxsRULLVLqB9gINkEQFO0U1ELP3Xm7Thab1O6BwzL37c6+eYxZQcFs1HLr5T8yrZXP5fKELr7gBS7iIU7w07MqbuIDvuIz7iRfOns4wgF+4xDLXtv37A6bni3hLd77OqGDbRzjI65FtZ6FTdRZSstC1+dRlqHCFVZ8rW7O8N3CUVIOLGzcj7IMFY6jdd1Cy2pdRxDa+BQ/cMuzKbSJjpayjV94giXPVi0M99rCsDWzDM0hPorQUd5wI8qOLGysH2jQK1HNdv2FmBv7/WPKOj7hpden0OXJXyBdrIVcJuax4c+C/8EP4awvkEhnxOwAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAAZCAYAAADOtSsxAAADJ0lEQVR4Xu2ZS6hNURjHP6G88s4joqRMlIEiIiUTAzKgZGCCGChhoJQyuQNJSUaSVylCEiOZiKJMDJgYIY8oyoBCHv/fXXvds8+6Z5+z99r7cFz7V//OPmvte863zn+tb31rX7Oaf46Z0iFpjzQ+6OsliPGU9X6chTgh3TdnwjLpq3Sg6Y7egDiJEXycQ4Jj0jtpgbkBvpDON93RGxAnMYKPc0gwXBqTXC+WPkuHB3p7B+L0+Dj/NsOkedLosCOW49IzaW7Y0UMwaB9nlYyUNoWNbSCOs9IncxOiNEukGWFjQdaFDRVDjNetfJyeo+bS7xFpStCXBUZNl8aZS4OsxNIGMONPJtezLT4FddsAYpycXBNnLOwlF6Rt0tigrwi5DBglTbNGDg3f8+PfkbZLm6Uz0u6kryixBrCcO81A4iTGjdaIsyjLpQfSU2lD0BdDRwMuSq+lN9IHaWtyjR4n95yWfgVam/QVJcaACdIl6a2035wZnhvSZXOTplWceSFtYNptaak1f0cZ2hqwyxpfxCtmrLdGCVdkAHkpagD1/FVzVcRC6aE5QzzEWOZMwuf2Sa/M5fmqyTSA9EJ+80ySHpkrmTBjUaIY+GxSGEaGYoWFbWhi/18OhpXmVxs/dDr1jZB+SmtSbbGQ5zk9k/f9WaIKMg0I8XUzS7ksbH53pZct9LFFG+IQxQ+aBbOe2c8q8MySnievVUHeJ/+zD5CKypLbgJ3WnZQTUjQFeVZI3615gjDzSU/tjIuBDMCPz36AIewPsbQ1gMHw4QyAgbCcPbRvSb2vilgD9tngCUJKKpP/88BBioIkthzNNMCnHPI8m/EPcxsRcIi5aW7WVU2sAVQo3gBmKNUQ76cO3NF9MGBv2NgG4nwvfTG3WtOPSvp3/3PSrUQY8cRcSUcOXD1wZ7XEGkC81PVXzBULxIgBVZWLVZKuIkM1rQSC53DjB+GrlyanKibWAA+D44jPo+cwJdXkYGXY0AEmB8v+m7QjaZtvLi+zYmu6DKmH/eiaNMfcyZeydVX6phL4LBCeS7LE6vvv4F+LB6V75p7zVPZsvaampqam5k/zG5z8nSMzJv6nAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAZCAYAAABQDyyRAAABuElEQVR4Xu2VPSiFURjHH6HkI4lIFEkiTDIQgzL4CEUxGA0MMpCE2egmixIDC8pqU4QJiyKlDBZ2ZcX/73nee9/3YHC9r1L3V79u53nOPec5555zrkiKFMosXHcsDvQQ2YGblqO9wXTypMFWeAHf4COcgJn+TmAI3oj22YZFwfTvGRcd/AGWOjnCovbkc2Gh0SdawAtscnLkCta7wTDhpJz8FXY6Oa56xomFThV8Et2FQSc3AnOdWOhwgmPRAjYsxpXzxM9bO3K2RAvYt3Y/PIGF8R4RsyRawCksh+ewPdAjYqYlcRXX4IroO/FneFfxWfRhcl/DyPHfhAwnR6ZgjegZqbDYsOjhzYeLsFZ0DB7gdLgLr60vF/jdQ/dBCbyHt24CZMEe0S97AxKemwNYCTtgt+gYpEB0JyetzdeWbca/hKvugo1uwkeb6CqId3Xn4tlEQaROtJhm0bG5c6uWSxqugreEtIgW0yD6E+TAQ7hg+VHRPzD+cbEY9uUCqi3/YzgQt5/vA+HnGVy2Np/wO1gmujtHcMyXu4QxOGCxpOAq/VeTE/F8EB66bF8uz2Ie7Bv5k57if/EO+89PGe2hQEkAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA3klEQVR4Xu2RPQ5BQRRGr0IjEZHYhEKlEq1GJ6LT6FiAlSgVVEoasQKl2gIsQKmQCL4vdzDue5m8oX0nOYU5482fSE4sA7iCc2PXdTtO26590ZDPxx5wAnuw5rrfTnDktQRluBedbOGfjqKtb1qCOjzDiw2gKTrOznlBuBJXPNgAxqKNO+bOg8xEJy9tEB1j45wgr2Nd4VT0cn1vrkcdq2oa+elYBdP4O9OxuAPuhJOHphE+ffSz85ktHMt0P2vRFTewZFoFbuECFk170xJ9JX7Edwc78J7S0u4wJ+dvnux2QnpaZwDfAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAlElEQVR4XmNgGHogHojnAvEsJAziRyErAgFzIA4B4qVA/B+In0H58siKkMEcBojC+egSyEAQiE8zQBRGo8mhAJAkSBFIMUgTTjCJAaIQZD1OQLS1mkD8Fog/AbE+mhwKINp9RFkLAiArvwKxMboEOgCZdhWIRdAlkAE/A0ThWiBmR5MDA08GiAJ0fACIeRDKRgE1AABrtiS5oX9iEgAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAuCAYAAACVmkVrAAAE60lEQVR4Xu3dT8gtYxwH8EcoSvn/LyRkIUpSpFihbFiwUcjSRpS/ZSWylSRJ6mbl70KhhMW7sLNAkYXUJRKFUhb+e77vzHPvnOfOvO99uW/K+Xzq1znnmXPmzMzq22+emSkFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4D91eK0Hah3RL5hxSa23+sE1ckKtG/rBBffUOrIfBADW11W1Pq/111hf1nq81nG13qj1R61fx/Hzxt80GbuwG7ttHJ+uL+uJw2q9Pb7fTdeU1W24cnXxPteV1e18cXydjqW+Gz/fP/xsxUatJ2qdXuvsWi/Venr6hTIsf70bi0dq/VCGdec1/3XUuOyjWveO7wEANiU0vN8PVj+V+cBza62L+sGJpfU9VeuUfnCXvFbr7lp7u/E4p9YrZdjOjdVFm2NfdGPx7FjT7le+1wLe17WOniyLfDedxX682VOG3/ZdylNrfVqGkAsAsBkWEhqe6xeUYTzhYeqYMoScPmQ06RLld0/2C8rQ/bq5H9wlCWxXl2FberfUurPsLLCdVevPWjdOxj6sdenkcy/L8l9LPinz2xc5fgmWAACbHbR00i7uxo8v82Hijlq/9YMTCWXf17qgX1CGMJeQclK/YBcksCUwfVMO7AY+U/YH1Y3VRYuBLbLs58nnrQJb9vXNMhzHJVlfjsecHL+50AsArKFMcs+cqZO78QSRuWCWTlxC0JIHy3A6dCmozIXD3dAC2wu1Hl5dtK/L908C2zTEJrDdXobjkf3KxRVNOpOZH7jUiczpzqzr1X7BKKH23X4QAFg/LVTMnbZLMJsLExtjzTmjDHPG5tbXJKRc3w9OZAL/dnUw8+BaYMs+5lRmLjKI9hr/NrDdV+vMyed03z4b3+e/p924XgvEc3MEI0FvaTsAgDWSsJDQ0J++THcsXbJ033obY83J6dCEo359U9sFtkOlBbbIf+aCh3S18tr828DWaxchxHaBLaeWtzs9vLQdAMAaSQdtLoCkQ5bx/oKD2BhrTuZsza1varvA1nfT5monHbbIlZr531xokGp2EtjanLecYo1c+ZnP09tvbIxjsV1gS1ibC8SNDhsAsGlvmQ9Ye8owPjf/KvOqloLE3jK/vql04NKJW9Luf7ZV5TTudqaBLd2sbNfLtc7d942dBbYWwC4fPyc45rsJqc20w5Z5epnXtiTrWjodGrkad+mCBABgDeTUYAscH5ehY5UnFyQkZPyXWr+P73tzV4meWFbXl87c3N36c+XkVhckHAr5j7YPuZFtbgKc/U0IbfPXsr9tez8Y30/3ITfLbd2882s9VIYbCGc9U7kR8Gnj++xvfpunP8TSVaL578x7y3dzq5ClbmFOKz/WDwIA6+OyMgSGVu02HO91461bNHVFGcLQVP+bdI/mbneRixLaXLLdktOt0215fhy/tgyhLFonbFobM2Otvi3z4TWnV3PsckPdBNW7yhB8m4fLgfP5+nXvXVm6Xx5lNb1AAgBgRxJU+nubHYyEtWP7wf+xzHPb6kkHS9qTDgAA/rGcxnu07LxT9lU/sAZ+LMMzW3cij/5ax2MFABximav1Tj+4hZvK8ID0dZOOYo7TwYbbzIXrn1kKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA/9DfImskBVu/9EUAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAyElEQVR4XmNgGCpAGIhPATEjugQxwAaIfzPg0BwPxHOBeBYSBvGjoPJFQPwPysYA5kAcAsRLgfg/ED+D8uUZILaBxE/DVWMBggwQBSDN0Uji+kD8AIiNkcQwAEjRJyD+yoCqEGTQDiDmRBLDACBFIFtBtoNcAQOToHJ4AUgRSPMcNHEhIOZGE0MBmkD8lgHibJDzSQK4nEwUADkVpHk+A46EgA9giyKiAbYoIgrwM0BsXQvE7GhyOIEnA0QTOj4AxDwIZaNgpAIA7W0rRP5XvgMAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHQAAAAZCAYAAADg8AqjAAAE9klEQVR4Xu2aW6huUxTHh1xycs+JRM458uKWI7fIg0KRS+I8yOXw5jycUh4ILzsSKQnnSXSSJEWR3A6xOZJ4QcSLQuJJnijJZfzOWGPvaXxjrbnWt769t47vV//aa8615pxjjjHHnGt9W2TOnDn/Tw5uNIR9VA+rPi3KDlPtX1zvbRyuOjAWdsAcHRELV4P3VBtiYYX1qi9VPxRll6veLq5nyclifa4V16heFgvavuDQ7dLzmXNVz6ueCLpPLJIeTOrQZh4uOEusrYybVU/JZBsHqc5U/ap6bulu40YZHhw1jlbtUp0YKyqwmh6QyfFf0tRfltShuApx5meqjaEcyGzxeebs+qYep74uPZx6rGqL6k7V36rHmuvzxdLeRWKT+7XqQ9V1YisIZzh0woqi0wwcTZu0/6fqnuaa+29V/aW6dOlug74xapapFxsXYmEP9lNdqHpfzIZvVFtleaUTeJl9ka/E5jJjndgzZR9cl0H9keoOaZ/nf4EjX5XJqIKLxSb9pFghFvUMdEesCNCuB4zj6ZZ0kvFGI4wdC5OAQ8fws9g8MB+RzL6SU6XdTof57eqjnOtOp7LcF6Xd4LvFBpsdeK4Si8psACUniLXB/Q7p9nPVpqKshNX7u+q8WDEFp6iOioUDYfzfimW1yHEyaV8Jc5stiJIbpLsPIGA4b7TN2R48MjKnEHmsXDqKECU7xTrAoC7YZ34Sc6yDAeydpLSMs1W/SXug9YX2Sd9jYQ66shhzmDnNF8whoTzypFgfL0j7nLAPdwXOHjwysqjAATgicyhH6U/E9s9yT824XybvY+89sriOHKP6TroN7AMr/J1YOBBsZQ7aggv7FiXPYj6HNZhL+rg9VhT4IZL+UtwpNMQmHPVaU5e9RlQbb3CDsgzQhWcHnIpzuzhNJseO7lV9LHZaLMuvtsd6Q9Azjltkso/bxDIJWSjjSskXRIR7OFOsjxUF1HHPYihfwp3CgCKeKugoc5o/2xa1DumBNsp025enpZ9DMxj/m7L8ejEtvrU82/wdqdnXx6G0W0u34D5BKRw8PDIitVMXqYxDS82hbOT0ke09NcY4lP0GJ4x99fFVwVxl1OzjVa/mUPrgnrY+nE6HeuTREIZHPPLaDj19VmiZ0ofC+BjXtA7dJTbGsbidHNIiZRZro88KvUAsS2Z9lHCw2i0tDvXIa4sMj7y2k50byumsjXKVD8Uni5ds3sG6eFxsrH01ZDyexbK9rY99fRxKH7X9E/ygmJ1p5CGxjr6QyRW4r1g6ZaCkrmzv4MRKw6zA7OMxp9gXZTkohuKHqbaX9S7IGtmYh8Dzp6t+FLOBOSkp7Xs31JUQjARltjeWfWxrrrtgJf8hYQGeI7a8Y9QSbYeqPkjq2iJsQfL3L4/qUt+rji9vquAfLeJnwRqbxGwYA86L40ceXJxuY13bCsORvG/HOoI8toGekcngcQjUX8S+PK0IfjDKUvZYmDw+ddXSbQkRzg8LXfv6WsDBKDtYDsFf42b1OTSFhl8R62TW4MyFWFiBT3xvyfjPfLOG8eyQekrtglVJNiQ4VhQGya8A1Z92BrBRdW0srMAh7SXJv9b8F+Dz406Z/jWKfZjtclWY5c9dBAYfBIZG86My/iPCSoJdZI/eP4EVbBA7nK4qd6muiIVTgME3xcK9BN4MHlGdESs6OEDs33TmzJkzZ87a8g/Hjz9DCwtaVwAAAABJRU5ErkJggg==>