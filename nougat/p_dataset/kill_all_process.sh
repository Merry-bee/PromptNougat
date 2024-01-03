#!/bin/bash

pkill python
pkill latexmk
pkill latex
kill -9 $(pgrep -f latexmlc)
kill -9 $(pgrep -f latexmk)
pkill kpsewhich
pkill rm
pkill cp
ps -ef