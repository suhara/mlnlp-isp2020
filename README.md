# Machine Learning and NLP: Advances and Applications

This repository hosts the course materials used for a 3-day seminar "Machine Learning and NLP: Advances and Applications" as part of Independent Study Period 2020 at [New College of Florida](https://www.ncf.edu/).

Note that the seminar was held in Jan 2020, and the content may be a little bit oudated (as of Feb 2022). Please also refer to a Fall 2021 full semester course ["CIS6930 Topics in Computing for Data Science"](https://github.com/suhara/cis6930-fall2021), which covers much wider (and a little bit newer) Deep Learning topics.


## Syllabus 

### Course Description

This 3-day course provides students with an opportunity to learn Machine Learning and Natural Language Processing (NLP) from basics to applications. The course covers some state-of-the-art NLP techniques including Deep Learning. Each day consists of a lecture and a hands-on session to help students learn how to apply those techniques to real-world applications. During the hands-on session, students will be given assignments to develop programming code in Python. Three days are too short to fully understand the concepts that are covered by the course and learn to apply those techniques to actual problems. Students are strongly encouraged to complete reading assignments before the lecture to be ready for the course assignments, and bring a lot of questions to the course. :)

### Learning Objectives
Students successfully completing the course will
- demonstrate the ability to apply machine learning and natural language processing techniques to various types of problems.
- demonstrate the ability to build their own machine learning models using Python libraries. 
- demonstrate the ability to read and understand research papers in ML and NLP.

### Course Outline

- Wed 1/22 Day 1: Machine Learning basics [[Slides]](slides/2020-01-22_day1_MLBasics.pdf)
    - Machine learning examples
    - Problem formulation
    - Evaluation and hyper-parameter tuning
    - Data Processing basics with pandas
    - Machine Learning with scikit-learn
    - Hands-on material: [[ipynb]](notebooks/ncf_isp2020_mlnlp_day1_student.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suhara/mlnlp-isp2020/blob/main/notebooks/ncf_isp2020_mlnlp_day1_student.ipynb)

- Thu 1/23 Day 2: NLP basics [[Slides]](slides/2020-01-23_day2_NLPBasics.pdf)
    - Unsupervised learning and visualization
    - Topic models
    - NLP basics with SpaCy and NLTK
    - Understanding NLP pipeline for feature extraction
    - Machine learning for NLP tasks (text classification, sequential tagging)
    - Hands-on material [[ipynb]](notebooks/ncf_isp2020_mlnlp_day2_student.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suhara/mlnlp-isp2020/blob/main/notebooks/ncf_isp2020_mlnlp_day2_student.ipynb)
    - Follow-up
        - Commonsense Reasoning (Winograd Schema Challenge)

- Fri 1/24 Day 3: Advanced techniques and applications [[Slides]](slides/2020-01-24_day3_AdvancedTechniques.pdf)
    - Basic Deep Learning techniques
    - Word embeddings
    - Advanced Deep Learning techniques for NLP
    - Problem formulation and applications to (non-)NLP tasks
    - Pre-training models: ELMo and BERT
    - Hands-on material: [[ipynb]](notebooks/ncf_isp2020_mlnlp_day3_student.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suhara/mlnlp-isp2020/blob/main/notebooks/ncf_isp2020_mlnlp_day3_student.ipynb)
    - Follow-up
        - The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time
        - Cross-lingual word/sentence embeddings
            - [MUSE](https://github.com/facebookresearch/MUSE) by FAIR
            - [LASER](https://github.com/facebookresearch/LASER) by FAIR
            - [Emu](https://arxiv.org/abs/1909.06731) (our paper)


### Reading Assignments & Recommendations:

The following online tutorials for students who are not familiar with the Python libraries used in the course. Each day will have a hands-on session that requires those libraries. Please do not expect to have enough time to learn how to use those libraries during the lecture.

- Pandas tutorials:
    - [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
- scikit-learn tutorials:
    - ["An introduction to machine learning with scikit-learn"](https://scikit-learn.org/stable/tutorial/index.html)
    - The other tutorials are also recommended
- gensim:
    - [Core tutorials](https://radimrehurek.com/gensim/auto_examples/index.html)
- spaCy:
    - [spaCy 101: Everything you need to know](https://spacy.io/usage/spacy-101)
- PyTorch:
    - [Deep Learning with PyTorch: A 60 Minutes Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

The following list is a good starting point. 

- Awesome - Most Cited Deep Learning Papers
    - [Natural Language Processing](https://github.com/terryum/awesome-deep-learning-papers#natural-language-processing--rnns)

The course will cover the following papers as examples of (non-NLP) applications (probably in Day 3.) Students who'd like to learn how to apply Deep Learning techniques to your own problems are encouraged to read the following papers.
- [1] A. Asai, S. Evensen, B. Golshan, A. Halevy, V. Li, A. Lopatenko, D. Stepanov, Y. Suhara, W.-C. Tan, Y. Xu, "HappyDB: A Corpus of 100,000 Crowdsourced Happy Moments" Proc LREC 18, 2018. [[Paper]](https://arxiv.org/abs/1801.07746) [[Dataset]](https://megagonlabs.github.io/HappyDB/)
- [2] S. Evensen, Y. Suhara, A. Halevy, V. Li, W.-C. Tan, S. Mumick, "Happiness Entailment: Automating Suggestions for Well-Being," Proc. ACII 2019, 2019. [[Paper]](https://arxiv.org/abs/1907.10036)
- [3] Y. Suhara, Y. Xu, A. Pentland, "DeepMood: Forecasting Depressed Mood Based on Self-Reported Histories via Recurrent Neural Networks," Proc. WWW '17, 2017.
[[Paper]](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p715.pdf)
- [4] N. Bhutani, Y. Suhara, W.-C. Tan, A. Halevy, H. V. Jagadish, "Open Information Extraction from Question-Answer Pairs," Proc. NAACL-HLT 2019, 2019. [[Paper]](https://aclanthology.org/N19-1239/)


### Computing Resources:

The course requires students to write code:
- Students are expected to have a personal computer at their disposal. Students should have a Python interpreter and the listed libraries installed on their machines.

The hands-on sessions will require the following Python libraries. Please install those libraries on your computer prior to the course. See also the reading assignment section for the recommended tutorials.
- pandas
- scikit-learn
- gensim
- spacy
- nltk
- torch (PyTorch)




