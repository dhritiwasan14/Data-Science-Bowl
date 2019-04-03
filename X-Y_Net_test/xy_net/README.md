# XY Network for Nuclear Segmentation in Multi-Tissue Histology Images

A dual branch network, where one branch predicts the nuclear pixels while the other branch predicts the X and Y distance of each pixel to the nearest nuclear centroids. <br />

[Link to paper](https://arxiv.org/abs/1812.06499)

## Repository Structure

* `src/` contains executable files used to run the model. Further information on running the code can be found in the corresponding directory.
* `src/metrics/` contains the evaluation code. Further information can be found in the corresponding directory.
* `data/` refers to information on the data used within the paper

## XY-Net

![](network.png)

(a) Overall network architecture <br />
(b) Residual unit <br />
(c) Dense decoder unit <br />
(d) Key


## Citation

If any part of this code is used, we appreciate appropriate citation to our paper: <br />

**Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Jin Tae Kwak, and Nasir Rajpoot. "XY Network for Nuclear Segmentation in Multi-Tissue Histology Images." arXiv preprint arXiv:1812.06499 (2018).** <br />

BibTex entry: <br />
```
@article{graham2018xy,
  title={XY Network for Nuclear Segmentation in Multi-Tissue Histology Images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:1812.06499},
  year={2018}
}
```

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


