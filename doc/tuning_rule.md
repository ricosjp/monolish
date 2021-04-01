# Development Policy for high performance {#tuning_rule}

1. Don't implement functions that clearly do not provide performance.
1. Don't implement functions that clearly do not provide performance.
1. Don't allocate memory in ways that users cannot anticipate.
1. Don't require users to change their programs due to changes in sparse matrix format.
1. Don't require users to change their programs due to changes in hardware architecture.
1. Don't require users to change programs due to changes in data types

## Don't implement functions that clearly do not provide performance.
説明を書く

## Don't implement functions that clearly do not provide performance.
説明を書く

## Don't allocate memory in ways that users cannot anticipate.
説明を書く

## Don't require users to change their programs due to changes in sparse matrix format.
The implementation of the matrix storage format in monolish has several features.

The matrix format has the following two attributes. 1. editable
1. editable
2. computable

A matrix storage format with editable attributes can reference and add matrix elements.

A matrix storage format with computable attribute can be used for computation and has functions such as transfer to GPU (see here for a list of computation functions).

In monolish, Dense format is considered as one of the sparse matrix formats.
Currently, there are four matrix storage formats available: Dense, COO, CRS, and LinearOperator.
The attributes of each matrix storage format are shown below.
- Dense: editable, computable
- COO: editable
- CRS: computable
- LinearOperator: computable

LinearOperator does not have all utility functions.

Matrices stored in COO format can be easily referenced, but the computation cannot be parallelized. COO does not have any attributes to compute.
The matrix stored in CRS format can be parallelized efficiently. CRS does not have the editable attribute.

CRS does not have the editable attribute; matrices stored in Dense format can be edited and computed efficiently.
LinearOperator is a special format. LinearOperator is a special format that can be created from any matrix storage format, but it does not have editable attribute.

The LinearOperator is a special format that can be created from all matrix storage formats, but it does not have an editable attribute. It will be implemented as computable when monolish implements ELL, JAD, and other storage formats.


## Don't require users to change their programs due to changes in hardware architecture.
説明を書く

## Don't require users to change programs due to changes in data types
説明を書く
