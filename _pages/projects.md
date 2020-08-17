---
layout: page
title: Projects
permalink: /projects
---

* this line must be ignored
{:toc}

## Object detection and robotics

During my stay at FEI University Center, I was responsible for integrating object detection techniques into a domestic robot. I worked with libraries such as OpenCV and the TensorFlow Object Detection API as well as ROS, the Robot Operating System. This section provides a centralized collection of resources I've created during my years working with computer vision and object detection.

Since you're here, take a look at the [set of Python scripts](https://github.com/douglasrizzo/detection_util_scripts) I provide to help in the creation of TFRecord files and label maps for the TensorFlow Object Detection API, as well as TXT annotation files for YOLOv3. I also authored a brief tutorial on how to train a detection model using the TensorFlow Object Detection API with the help of my scripts.

### dodo detector (Python)

[dodo detector](http://douglasrizzo.com.br/dodo_detector/) is a Python package that encapsulates OpenCV object detection via keypoint detection and feature matching, as well as the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) in a single place. See a simple tutorial [here](https://gist.github.com/douglasrizzo/fd4cff7cdf53b3ad08d67f736e5017ea). [[Relevant paper]](https://www.researchgate.net/publication/338032150_CAPTION_Correction_by_Analyses_POS-Tagging_and_Interpretation_of_Objects_using_only_Nouns).

<iframe width="560" height="315" src="https://www.youtube.com/embed/Py6_qG52EYQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### dodo detector ros (ROS)

I am also the creator of [dodo detector ros](https://github.com/douglasrizzo/dodo_detector_ros), a ROS package that allows dodo detector to interface with USB cameras as well as both Kinects v1 and v2, in order to detect objects and place them in a 3D environment using the Kinect point cloud. [[Relevant paper]](https://www.researchgate.net/publication/333931333_HERA_Home_Environment_Robot_Assistant?_sg=GeiJpHAg-qFfldnKYUJofw09SmBojDPMoOVXAXBtRN0PQoe-1N-CM7ry2q89Gq0zfcwUusFYgBCG1U3dN-KoIGfndqnR9tazsZ9_gafb.7OO3N70IPnsb377if8wOMVhPMKJnucTmYH7hn34kpeBcKn_KwIOVF1m28fGLgwgO06jL6mvZR1RcBnDIYMAvwQ).

<iframe width="560" height="315" src="https://www.youtube.com/embed/fXJYmJOaSxQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Computerized adaptive testing

For years, I have been a member of the [Study and Research Group in Educational Assessment](http://dgp.cnpq.br/dgp/espelhogrupo/2558665306599960) (a free translation of _Grupo de Estudos e Pesquisas em Avaliação Educacional, Gepave_), where I specialized in the study of [Item Response Theory](https://en.wikipedia.org/wiki/Item_response_theory) and [computerized adaptive tests](https://en.wikipedia.org/wiki/Computerized_adaptive_testing). While I am not active as of late, I had the opportunity to be a part in a few projects, listed below.

### catsim

<center>
<img src="https://douglasrizzo.com.br/catsim/_static/logo.svg" width="40%"/>
</center>

A Python package that simulates a set of examinees taking a computerized adaptive test. There are different options for initialization, selection and proficiency estimation methods as well as stopping criteria for the test. Useful for studying item exposure and can also be used to power other applications. Documentation [here](http://douglasrizzo.com.br/catsim/). ArXiv paper [here](https://arxiv.org/abs/1707.03012).

### jCAT

Publication in [English](https://www.researchgate.net/publication/326803834_How_to_build_a_Computerized_Adaptive_Test_with_free_software_and_pedagogical_relevance) and [Portuguese](https://www.researchgate.net/publication/327704465_Teste_Adaptativo_Informatizado_Como_Recurso_Tecnologico_para_Alfabetizacao_Inicial).

jCAT is a Java EE web application whose purpose is to apply both an electronic version and a computerized adaptive test of [Provinha Brasil](http://provinhabrasil.inep.gov.br/provinhabrasil/), a nation-wide educational evaluation for Brazilian students in the second year of basic school.

<center>
<img src="/images/jcat-item.png" width="48%"/> <img src="/images/jcat-relatorio.png" width="48%"/>
</center>

## Miscellaneous Projects

### FEI LaTeX class

a LaTeX class used by [FEI University Center](http://www.fei.edu.br/) students to author their term papers, masters dissertations and doctoral theses under the institution typographical rules. The document class, a `.tex` template file and the PDF documentation are available on [CTAN](https://ctan.org/pkg/fei). There is also a [wiki](https://github.com/douglasrizzo/Classe-Latex-FEI/wiki) for miscellanoues tips and a [Google group](https://groups.google.com/forum/#!forum/grupo-latex-fei) (in Portuguese) for me to communicate with users.

### Algorithm Implementations

These were implemented in C++:

* [machine learning and reinforcement learning algorithms](http://douglasrizzo.com.br/machine_learning/)
* [numerical analysis procedures](http://douglasrizzo.com.br/numerical_analysis/)
* [full-fledged matrix class](http://douglasrizzo.com.br/matrix/)

They may not be directly useful in third-party projects, as they are not as optimized as their commercial counterparts, but they are nonetheless well documented for those interested in learning.
