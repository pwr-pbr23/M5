#!/bin/bash

codeql database upgrade placeholderdb

codeql query run --database=placeholderdb --output=temp.bqrs knownUnknown_groundTruth.ql
codeql bqrs decode --format=csv temp.bqrs | tail -n+2 > knownUnknown.csv
rm temp.bqrs

codeql query run --database=placeholderdb --output=temp.bqrs broken_groundTruth.ql
codeql bqrs decode --format=csv temp.bqrs | tail -n+2 > broken.csv
rm temp.bqrs

codeql query run --database=placeholderdb --output=temp.bqrs correct_groundTruth.ql
codeql bqrs decode --format=csv temp.bqrs | tail -n+2 > correct.csv
rm temp.bqrs


