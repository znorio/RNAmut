# RNAmut
## Description
To gain a better understanding of tumor heterogeneity, it is important to reliably detect cell-level mutations. This can be done using single cell DNA data, for example https://github.com/cbg-ethz/SCITE. Based on this work, but with focus on RNA instead of DNA data, RNAmut aims to determine single cell mutations from alternative and reference nucleotide read counts. To achieve this, it makes use of the phylogenetic relation between single cells. For more details see [*main.py*](https://github.com/znorio/RNAmut/blob/master/main.py). \
Two read count files were extracted from single-cell RNA sequencing data by following the steps in section 5 https://github.com/nghiavtr/SCmut. 
These files, serving as examples, can be found in the [*Data*](https://github.com/znorio/RNAmut/tree/master/Data) folder.
