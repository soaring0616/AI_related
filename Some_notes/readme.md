### Weaviate_v4_update.ipynb
改版測試紀錄

### Some_Mathematical_Intuitions_and_Definitions_in_Calculus__Probability_and_Linear_algebra.pdf
一些機器學習(或是應用數學)中，對於微積分、線性代數、和機率重要的notion
* 微積分：
  * 連續: 說明 function 所處之 domain 的完善
  * 極限: 說明 the availablity of differentiation
  * 收斂: 說明 the availablity of integration
* 線性代數：
  * 向量空間: 說明線性轉換所執行的 domain 需保持一些算數的基本性質(分配律、交換律 etc)
  * 線性轉換: 了解什麼稱為線性轉換(數乘、疊加)
  * 矩陣與線性轉換的關係: 需要基底來將抽象化的線性轉換建構出數值表現的矩陣
* 機率:
  * measure & measurable
  * probability as a measure function
  * random variable as a mapping function 

### A_study_of_attention_mechanics_in_the_industry_of_artificial_intelligence_202410.pdf

* 最受益的是幾個點：
1. Attention 的q和k的內積其實是傳統統計學的東西，再根本的含義其實是期望值（但精確一點應該是統計上的 **kernel**） -> [kernel、期望值與條件機率](https://chatgpt.com/share/67b40277-08c8-800a-be0d-067d2a56ccd2)
2. 從kernel採納了Gaussian filter，softmax自然而然就呼之欲出
3. 就是對於masked, self-attention, multi-head attention 他們的計算過程比較熟悉一點（裏面也有context vector計算）
4.  Additive Attention 是在計算成本的考量下，可作為替代的方案；

* 接下來是在找 attention 時找到的資源，雖然不方便放在正式文件，但還是可以用來 presentation 的：
  * 以下是主要參考的文獻
    * https://classic.d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html
    * https://en.wikipedia.org/wiki/Attention_(machine_learning)
    * https://arxiv.org/abs/2204.13154
    * https://arxiv.org/pdf/1409.0473

  * 接下來是有參考價值的資源
    * https://www.kaggle.com/code/lianghsunhuang/attention-mechanism
    * https://lilianweng.github.io/posts/2018-06-24-attention/#summary
    * https://en.wikipedia.org/wiki/Neural_machine_translation
    * https://socialsci.libretexts.org/Bookshelves/Psychology/Cognitive_Psychology/Cognitive_Psychology_(Andrade_and_Walker)/11%3A_Attention/11.02%3A_History_of_Attention
    * https://distill.pub/2016/augmented-rnns/#neural-turing-machines
    * https://jaketae.github.io/study/seq2seq-attention/

  * 最後就是感謝 NotebookLM 和 Chatgpt；
Chatgpt 比較容易發散，但 NotebookLM 的好處之一就是「僅限於」使用者所提供的資源；
故此，使用順序是先透過 NotebookLM ，再讓 Chatgpt 來發揮。

Umm，或許這也是之後 AI 使用者所需要培養的素養吧 😎
