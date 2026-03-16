Here is my current research journey, first I started with Literature Review and searching online for similar notebooks for my task. I found some examples and made docs for lineage "v0x" line of experimentation. This was mostly done on documentation with the help of gemini deep research, chatgpt and github copilot to write the docs and codex 5.4 to audit those docs. then making new docs. with little to no code until v06_5 and then v08 and further. I was thinking that first I should get the documentation right and then make notebook from it but this approach was not the best. 

Then the further versions added too many things at once leading to a catastrophic failure,

Lesson - Adding too many things from research papers or ai suggestions without experimentation leads to failures.

next I found a notebook on kaggle "https://www.kaggle.com/code/gpsattendanc/image-detection-with-mask" titled "Image_detection_with_mask-". Initially I was delighted to have found it as it was doing my assignment task of Image Tampering Detection and Localisation. So i discontinued the earlier experiment line and started the lineage of "vK.x.x" series. I followed the approach of the found notebook and ran my training. So I added many cherry on tops things by keeping the architecture same. But upon audit of my notebook runs I found that I should make my pipeline data leakage proof so I did. This lead to very decrease in results. I was not able to understand the issue so upon consulting ai, it guided me towards the fact my model was training from scratch on a dataset of approx 10k images. So it was learning edges, corners, etc only. Hence it suggested me to use a pretrained model so I used ResNet 34. But my results were very less than the original notebook. This lead to do a ai audit on the original notebook. Upon which I found that the notebook run is suffering from data leakage (reason) and the complete approach is fundamentally flawed. So my lineage of "vK.x.x" where K = Kaggle was also discontinued.

next back to the literature review phase so the review and analysis of many research papers was done by AI and documented then the best for my approach was selected to recreate. Upon finding the original code unavailable, we had to use the research paper in order to recreate the code.
I took to recreating a research paper architecture to server a baseline so I choose "ETASR_9593.pdf" and did some ablation study on the lineage "vR.x.x." , where R = Research. But I soon found out that this approach is limited to classification only and is unsuitable for localisation task. Thus I also discontinued the lineage "vR.x.x". So this lineage was training from scratch 

So for the next Iineage evolution I decided to use pretrained weights and made lineage "vR.P.x.x", where R = Research and P = Pretrained.
This time I learned from my failure of adding too many things at once. So I decided to do ablation study by adding only one component per version. 
I also tracked the experiments of this lineage using weights and biases

Upon ablation I also experimented with ResNet50 and EfficientNet but results indicated that ResNet32 was superior for my use case

Moroever, ELA was chosen instead of RGB image to detec tampering.

[Then details about this lineage and it's results]

About the Dataset, I searached many datasets, but the best one was the dataset used the kaggle notebook which had ground truth masks added to the data of sunnyhaze github repo which itself improved upon CASIA2

So the dataset which was finally used was "https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization/data" 

because the dataset of sunnyhaze available at "https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset" lacked many ground truths

Among the availabel datasets this was the biggest one and runnable on compute restrained environment of t4


I also found "FakeShield" and tried to implement a pruned version but it was too complicated but it was stil very comlex and time taking so I discontinued that version too