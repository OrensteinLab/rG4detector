#!/bin/bash

`bedtools getfasta -fi big_files/$1 -bed $2 -fo $3.fa -s`
`sed '1d; n; d' $3.fa > $3.txt `

