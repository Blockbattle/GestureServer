
Ö
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.7.02v2.7.0-0-gc256c071bb28çÈ
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	à* 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes
:	à*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
ª
'simple_rnn_50/simple_rnn_cell_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'simple_rnn_50/simple_rnn_cell_50/kernel
£
;simple_rnn_50/simple_rnn_cell_50/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_50/simple_rnn_cell_50/kernel*
_output_shapes

:@*
dtype0
¾
1simple_rnn_50/simple_rnn_cell_50/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*B
shared_name31simple_rnn_50/simple_rnn_cell_50/recurrent_kernel
·
Esimple_rnn_50/simple_rnn_cell_50/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_50/simple_rnn_cell_50/recurrent_kernel*
_output_shapes

:@@*
dtype0
¢
%simple_rnn_50/simple_rnn_cell_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%simple_rnn_50/simple_rnn_cell_50/bias

9simple_rnn_50/simple_rnn_cell_50/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_50/simple_rnn_cell_50/bias*
_output_shapes
:@*
dtype0
«
'simple_rnn_51/simple_rnn_cell_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@à*8
shared_name)'simple_rnn_51/simple_rnn_cell_51/kernel
¤
;simple_rnn_51/simple_rnn_cell_51/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_51/simple_rnn_cell_51/kernel*
_output_shapes
:	@à*
dtype0
À
1simple_rnn_51/simple_rnn_cell_51/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àà*B
shared_name31simple_rnn_51/simple_rnn_cell_51/recurrent_kernel
¹
Esimple_rnn_51/simple_rnn_cell_51/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_51/simple_rnn_cell_51/recurrent_kernel* 
_output_shapes
:
àà*
dtype0
£
%simple_rnn_51/simple_rnn_cell_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*6
shared_name'%simple_rnn_51/simple_rnn_cell_51/bias

9simple_rnn_51/simple_rnn_cell_51/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_51/simple_rnn_cell_51/bias*
_output_shapes	
:à*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	à*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes
:	à*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0
¸
.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*?
shared_name0.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/m
±
BAdam/simple_rnn_50/simple_rnn_cell_50/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/m*
_output_shapes

:@*
dtype0
Ì
8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m
Å
LAdam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
°
,Adam/simple_rnn_50/simple_rnn_cell_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/simple_rnn_50/simple_rnn_cell_50/bias/m
©
@Adam/simple_rnn_50/simple_rnn_cell_50/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_50/simple_rnn_cell_50/bias/m*
_output_shapes
:@*
dtype0
¹
.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@à*?
shared_name0.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/m
²
BAdam/simple_rnn_51/simple_rnn_cell_51/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/m*
_output_shapes
:	@à*
dtype0
Î
8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àà*I
shared_name:8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m
Ç
LAdam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m* 
_output_shapes
:
àà*
dtype0
±
,Adam/simple_rnn_51/simple_rnn_cell_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*=
shared_name.,Adam/simple_rnn_51/simple_rnn_cell_51/bias/m
ª
@Adam/simple_rnn_51/simple_rnn_cell_51/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_51/simple_rnn_cell_51/bias/m*
_output_shapes	
:à*
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	à*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes
:	à*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0
¸
.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*?
shared_name0.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/v
±
BAdam/simple_rnn_50/simple_rnn_cell_50/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/v*
_output_shapes

:@*
dtype0
Ì
8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v
Å
LAdam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
°
,Adam/simple_rnn_50/simple_rnn_cell_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/simple_rnn_50/simple_rnn_cell_50/bias/v
©
@Adam/simple_rnn_50/simple_rnn_cell_50/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_50/simple_rnn_cell_50/bias/v*
_output_shapes
:@*
dtype0
¹
.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@à*?
shared_name0.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/v
²
BAdam/simple_rnn_51/simple_rnn_cell_51/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/v*
_output_shapes
:	@à*
dtype0
Î
8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àà*I
shared_name:8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v
Ç
LAdam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v* 
_output_shapes
:
àà*
dtype0
±
,Adam/simple_rnn_51/simple_rnn_cell_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*=
shared_name.,Adam/simple_rnn_51/simple_rnn_cell_51/bias/v
ª
@Adam/simple_rnn_51/simple_rnn_cell_51/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_51/simple_rnn_cell_51/bias/v*
_output_shapes	
:à*
dtype0

NoOpNoOp
ü7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*·7
value­7Bª7 B£7

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
Ð
&iter

'beta_1

(beta_2
	)decay
*learning_rate mn!mo+mp,mq-mr.ms/mt0mu vv!vw+vx,vy-vz.v{/v|0v}
8
+0
,1
-2
.3
/4
05
 6
!7
8
+0
,1
-2
.3
/4
05
 6
!7
 
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
	regularization_losses
 
~

+kernel
,recurrent_kernel
-bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
 

+0
,1
-2

+0
,1
-2
 
¹

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
~

.kernel
/recurrent_kernel
0bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
 

.0
/1
02

.0
/1
02
 
¹

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
"	variables
#trainable_variables
$regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'simple_rnn_50/simple_rnn_cell_50/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1simple_rnn_50/simple_rnn_cell_50/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_50/simple_rnn_cell_50/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'simple_rnn_51/simple_rnn_cell_51/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1simple_rnn_51/simple_rnn_cell_51/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_51/simple_rnn_cell_51/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

Y0
Z1
 
 

+0
,1
-2

+0
,1
-2
 
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
6	variables
7trainable_variables
8regularization_losses
 
 

0
 
 
 
 
 
 
 
 

.0
/1
02

.0
/1
02
 
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	etotal
	fcount
g	variables
h	keras_api
D
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

i0
j1

l	variables
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_50/simple_rnn_cell_50/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_51/simple_rnn_cell_51/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_50/simple_rnn_cell_50/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_51/simple_rnn_cell_51/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

#serving_default_simple_rnn_50_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
÷
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_50_input'simple_rnn_50/simple_rnn_cell_50/kernel%simple_rnn_50/simple_rnn_cell_50/bias1simple_rnn_50/simple_rnn_cell_50/recurrent_kernel'simple_rnn_51/simple_rnn_cell_51/kernel%simple_rnn_51/simple_rnn_cell_51/bias1simple_rnn_51/simple_rnn_cell_51/recurrent_kerneldense_25/kerneldense_25/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_190402
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;simple_rnn_50/simple_rnn_cell_50/kernel/Read/ReadVariableOpEsimple_rnn_50/simple_rnn_cell_50/recurrent_kernel/Read/ReadVariableOp9simple_rnn_50/simple_rnn_cell_50/bias/Read/ReadVariableOp;simple_rnn_51/simple_rnn_cell_51/kernel/Read/ReadVariableOpEsimple_rnn_51/simple_rnn_cell_51/recurrent_kernel/Read/ReadVariableOp9simple_rnn_51/simple_rnn_cell_51/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOpBAdam/simple_rnn_50/simple_rnn_cell_50/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_50/simple_rnn_cell_50/bias/m/Read/ReadVariableOpBAdam/simple_rnn_51/simple_rnn_cell_51/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_51/simple_rnn_cell_51/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpBAdam/simple_rnn_50/simple_rnn_cell_50/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_50/simple_rnn_cell_50/bias/v/Read/ReadVariableOpBAdam/simple_rnn_51/simple_rnn_cell_51/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_51/simple_rnn_cell_51/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_192220

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_25/kerneldense_25/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'simple_rnn_50/simple_rnn_cell_50/kernel1simple_rnn_50/simple_rnn_cell_50/recurrent_kernel%simple_rnn_50/simple_rnn_cell_50/bias'simple_rnn_51/simple_rnn_cell_51/kernel1simple_rnn_51/simple_rnn_cell_51/recurrent_kernel%simple_rnn_51/simple_rnn_cell_51/biastotalcounttotal_1count_1Adam/dense_25/kernel/mAdam/dense_25/bias/m.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/m8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m,Adam/simple_rnn_50/simple_rnn_cell_50/bias/m.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/m8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m,Adam/simple_rnn_51/simple_rnn_cell_51/bias/mAdam/dense_25/kernel/vAdam/dense_25/bias/v.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/v8Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v,Adam/simple_rnn_50/simple_rnn_cell_50/bias/v.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/v8Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v,Adam/simple_rnn_51/simple_rnn_cell_51/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_192329ù 
Ð
¾
-sequential_25_simple_rnn_50_while_cond_188831T
Psequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_loop_counterZ
Vsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_maximum_iterations1
-sequential_25_simple_rnn_50_while_placeholder3
/sequential_25_simple_rnn_50_while_placeholder_13
/sequential_25_simple_rnn_50_while_placeholder_2V
Rsequential_25_simple_rnn_50_while_less_sequential_25_simple_rnn_50_strided_slice_1l
hsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_cond_188831___redundant_placeholder0l
hsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_cond_188831___redundant_placeholder1l
hsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_cond_188831___redundant_placeholder2l
hsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_cond_188831___redundant_placeholder3.
*sequential_25_simple_rnn_50_while_identity
Ò
&sequential_25/simple_rnn_50/while/LessLess-sequential_25_simple_rnn_50_while_placeholderRsequential_25_simple_rnn_50_while_less_sequential_25_simple_rnn_50_strided_slice_1*
T0*
_output_shapes
: 
*sequential_25/simple_rnn_50/while/IdentityIdentity*sequential_25/simple_rnn_50/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_25_simple_rnn_50_while_identity3sequential_25/simple_rnn_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ò
¾
-sequential_25_simple_rnn_51_while_cond_188940T
Psequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_loop_counterZ
Vsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_maximum_iterations1
-sequential_25_simple_rnn_51_while_placeholder3
/sequential_25_simple_rnn_51_while_placeholder_13
/sequential_25_simple_rnn_51_while_placeholder_2V
Rsequential_25_simple_rnn_51_while_less_sequential_25_simple_rnn_51_strided_slice_1l
hsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_cond_188940___redundant_placeholder0l
hsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_cond_188940___redundant_placeholder1l
hsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_cond_188940___redundant_placeholder2l
hsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_cond_188940___redundant_placeholder3.
*sequential_25_simple_rnn_51_while_identity
Ò
&sequential_25/simple_rnn_51/while/LessLess-sequential_25_simple_rnn_51_while_placeholderRsequential_25_simple_rnn_51_while_less_sequential_25_simple_rnn_51_strided_slice_1*
T0*
_output_shapes
: 
*sequential_25/simple_rnn_51/while/IdentityIdentity*sequential_25/simple_rnn_51/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_25_simple_rnn_51_while_identity3sequential_25/simple_rnn_51/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
Ü	
Æ
.__inference_sequential_25_layer_call_fn_190423

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:	@à
	unknown_3:	à
	unknown_4:
àà
	unknown_5:	à
	unknown_6:
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_189892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
×
"__inference__traced_restore_192329
file_prefix3
 assignvariableop_dense_25_kernel:	à.
 assignvariableop_1_dense_25_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: L
:assignvariableop_7_simple_rnn_50_simple_rnn_cell_50_kernel:@V
Dassignvariableop_8_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel:@@F
8assignvariableop_9_simple_rnn_50_simple_rnn_cell_50_bias:@N
;assignvariableop_10_simple_rnn_51_simple_rnn_cell_51_kernel:	@àY
Eassignvariableop_11_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel:
ààH
9assignvariableop_12_simple_rnn_51_simple_rnn_cell_51_bias:	à#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: =
*assignvariableop_17_adam_dense_25_kernel_m:	à6
(assignvariableop_18_adam_dense_25_bias_m:T
Bassignvariableop_19_adam_simple_rnn_50_simple_rnn_cell_50_kernel_m:@^
Lassignvariableop_20_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_m:@@N
@assignvariableop_21_adam_simple_rnn_50_simple_rnn_cell_50_bias_m:@U
Bassignvariableop_22_adam_simple_rnn_51_simple_rnn_cell_51_kernel_m:	@à`
Lassignvariableop_23_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_m:
ààO
@assignvariableop_24_adam_simple_rnn_51_simple_rnn_cell_51_bias_m:	à=
*assignvariableop_25_adam_dense_25_kernel_v:	à6
(assignvariableop_26_adam_dense_25_bias_v:T
Bassignvariableop_27_adam_simple_rnn_50_simple_rnn_cell_50_kernel_v:@^
Lassignvariableop_28_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_v:@@N
@assignvariableop_29_adam_simple_rnn_50_simple_rnn_cell_50_bias_v:@U
Bassignvariableop_30_adam_simple_rnn_51_simple_rnn_cell_51_kernel_v:	@à`
Lassignvariableop_31_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_v:
ààO
@assignvariableop_32_adam_simple_rnn_51_simple_rnn_cell_51_bias_v:	à
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ê
valueÀB½"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_7AssignVariableOp:assignvariableop_7_simple_rnn_50_simple_rnn_cell_50_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_8AssignVariableOpDassignvariableop_8_simple_rnn_50_simple_rnn_cell_50_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_50_simple_rnn_cell_50_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_10AssignVariableOp;assignvariableop_10_simple_rnn_51_simple_rnn_cell_51_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_11AssignVariableOpEassignvariableop_11_simple_rnn_51_simple_rnn_cell_51_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_51_simple_rnn_cell_51_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_25_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_25_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_19AssignVariableOpBassignvariableop_19_adam_simple_rnn_50_simple_rnn_cell_50_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_20AssignVariableOpLassignvariableop_20_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_50_simple_rnn_cell_50_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_simple_rnn_51_simple_rnn_cell_51_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_23AssignVariableOpLassignvariableop_23_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_51_simple_rnn_cell_51_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_25_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_25_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_27AssignVariableOpBassignvariableop_27_adam_simple_rnn_50_simple_rnn_cell_50_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_28AssignVariableOpLassignvariableop_28_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_simple_rnn_50_simple_rnn_cell_50_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_30AssignVariableOpBassignvariableop_30_adam_simple_rnn_51_simple_rnn_cell_51_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_31AssignVariableOpLassignvariableop_31_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_simple_rnn_51_simple_rnn_cell_51_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü

I__inference_sequential_25_layer_call_and_return_conditional_losses_190348
simple_rnn_50_input&
simple_rnn_50_190326:@"
simple_rnn_50_190328:@&
simple_rnn_50_190330:@@'
simple_rnn_51_190334:	@à#
simple_rnn_51_190336:	à(
simple_rnn_51_190338:
àà"
dense_25_190342:	à
dense_25_190344:
identity¢ dense_25/StatefulPartitionedCall¢%simple_rnn_50/StatefulPartitionedCall¢%simple_rnn_51/StatefulPartitionedCall°
%simple_rnn_50/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_50_inputsimple_rnn_50_190326simple_rnn_50_190328simple_rnn_50_190330*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189733ë
dropout_28/PartitionedCallPartitionedCall.simple_rnn_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_189746½
%simple_rnn_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0simple_rnn_51_190334simple_rnn_51_190336simple_rnn_51_190338*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189859è
dropout_29/PartitionedCallPartitionedCall.simple_rnn_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189872
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0dense_25_190342dense_25_190344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_189885x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
NoOpNoOp!^dense_25/StatefulPartitionedCall&^simple_rnn_50/StatefulPartitionedCall&^simple_rnn_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2N
%simple_rnn_50/StatefulPartitionedCall%simple_rnn_50/StatefulPartitionedCall2N
%simple_rnn_51/StatefulPartitionedCall%simple_rnn_51/StatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input

ë
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192019

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
!
Ø
while_body_189243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_50_189265_0:@/
!while_simple_rnn_cell_50_189267_0:@3
!while_simple_rnn_cell_50_189269_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_50_189265:@-
while_simple_rnn_cell_50_189267:@1
while_simple_rnn_cell_50_189269:@@¢0while/simple_rnn_cell_50/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
0while/simple_rnn_cell_50/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_50_189265_0!while_simple_rnn_cell_50_189267_0!while_simple_rnn_cell_50_189269_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189187â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_50/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_50/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_50_189265!while_simple_rnn_cell_50_189265_0"D
while_simple_rnn_cell_50_189267!while_simple_rnn_cell_50_189267_0"D
while_simple_rnn_cell_50_189269!while_simple_rnn_cell_50_189269_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_50/StatefulPartitionedCall0while/simple_rnn_cell_50/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ý
d
F__inference_dropout_29_layer_call_and_return_conditional_losses_191942

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
£@
Â
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189733

inputsC
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189667*
condR
while_cond_189666*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
ª
while_cond_189079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189079___redundant_placeholder04
0while_while_cond_189079___redundant_placeholder14
0while_while_cond_189079___redundant_placeholder24
0while_while_cond_189079___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

ï
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192098

inputs
states_01
matmul_readvariableop_resource:	@à.
biasadd_readvariableop_resource:	à4
 matmul_1_readvariableop_resource:
àà
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàH
TanhTanhadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
states/0
Ú
ª
while_cond_191229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191229___redundant_placeholder04
0while_while_cond_191229___redundant_placeholder14
0while_while_cond_191229___redundant_placeholder24
0while_while_cond_191229___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

í
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189367

inputs

states1
matmul_readvariableop_resource:	@à.
biasadd_readvariableop_resource:	à4
 matmul_1_readvariableop_resource:
àà
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàH
TanhTanhadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_namestates
L
²
__inference__traced_save_192220
file_prefix.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_simple_rnn_50_simple_rnn_cell_50_kernel_read_readvariableopP
Lsavev2_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_50_simple_rnn_cell_50_bias_read_readvariableopF
Bsavev2_simple_rnn_51_simple_rnn_cell_51_kernel_read_readvariableopP
Lsavev2_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_51_simple_rnn_cell_51_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_50_simple_rnn_cell_50_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_50_simple_rnn_cell_50_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_51_simple_rnn_cell_51_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_51_simple_rnn_cell_51_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_50_simple_rnn_cell_50_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_50_simple_rnn_cell_50_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_51_simple_rnn_cell_51_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_51_simple_rnn_cell_51_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¡
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ê
valueÀB½"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_simple_rnn_50_simple_rnn_cell_50_kernel_read_readvariableopLsavev2_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_read_readvariableop@savev2_simple_rnn_50_simple_rnn_cell_50_bias_read_readvariableopBsavev2_simple_rnn_51_simple_rnn_cell_51_kernel_read_readvariableopLsavev2_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_read_readvariableop@savev2_simple_rnn_51_simple_rnn_cell_51_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableopIsavev2_adam_simple_rnn_50_simple_rnn_cell_50_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_50_simple_rnn_cell_50_bias_m_read_readvariableopIsavev2_adam_simple_rnn_51_simple_rnn_cell_51_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_51_simple_rnn_cell_51_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableopIsavev2_adam_simple_rnn_50_simple_rnn_cell_50_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_50_simple_rnn_cell_50_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_50_simple_rnn_cell_50_bias_v_read_readvariableopIsavev2_adam_simple_rnn_51_simple_rnn_cell_51_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_51_simple_rnn_cell_51_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_51_simple_rnn_cell_51_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesô
ñ: :	à:: : : : : :@:@@:@:	@à:
àà:à: : : : :	à::@:@@:@:	@à:
àà:à:	à::@:@@:@:	@à:
àà:à: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	à: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:%!

_output_shapes
:	@à:&"
 
_output_shapes
:
àà:!

_output_shapes	
:à:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	à: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	@à:&"
 
_output_shapes
:
àà:!

_output_shapes	
:à:%!

_output_shapes
:	à: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	@à:& "
 
_output_shapes
:
àà:!!

_output_shapes	
:à:"

_output_shapes
: 
Ú
ª
while_cond_191341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191341___redundant_placeholder04
0while_while_cond_191341___redundant_placeholder14
0while_while_cond_191341___redundant_placeholder24
0while_while_cond_191341___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ê,
Ù
while_body_191861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_189666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189666___redundant_placeholder04
0while_while_cond_189666___redundant_placeholder14
0while_while_cond_189666___redundant_placeholder24
0while_while_cond_189666___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ú
ª
while_cond_189242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189242___redundant_placeholder04
0while_while_cond_189242___redundant_placeholder14
0while_while_cond_189242___redundant_placeholder24
0while_while_cond_189242___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ã9
õ
simple_rnn_50_while_body_1907198
4simple_rnn_50_while_simple_rnn_50_while_loop_counter>
:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations#
simple_rnn_50_while_placeholder%
!simple_rnn_50_while_placeholder_1%
!simple_rnn_50_while_placeholder_27
3simple_rnn_50_while_simple_rnn_50_strided_slice_1_0s
osimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@V
Hsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@[
Isimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@ 
simple_rnn_50_while_identity"
simple_rnn_50_while_identity_1"
simple_rnn_50_while_identity_2"
simple_rnn_50_while_identity_3"
simple_rnn_50_while_identity_45
1simple_rnn_50_while_simple_rnn_50_strided_slice_1q
msimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource:@T
Fsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource:@Y
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp¢>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
Esimple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7simple_rnn_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_50_while_placeholderNsimple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ä
<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0ï
-simple_rnn_50/while/simple_rnn_cell_50/MatMulMatMul>simple_rnn_50/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0ë
.simple_rnn_50/while/simple_rnn_cell_50/BiasAddBiasAdd7simple_rnn_50/while/simple_rnn_cell_50/MatMul:product:0Esimple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ö
/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1MatMul!simple_rnn_50_while_placeholder_2Fsimple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ù
*simple_rnn_50/while/simple_rnn_cell_50/addAddV27simple_rnn_50/while/simple_rnn_cell_50/BiasAdd:output:09simple_rnn_50/while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+simple_rnn_50/while/simple_rnn_cell_50/TanhTanh.simple_rnn_50/while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
8simple_rnn_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_50_while_placeholder_1simple_rnn_50_while_placeholder/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ[
simple_rnn_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_50/while/addAddV2simple_rnn_50_while_placeholder"simple_rnn_50/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_50/while/add_1AddV24simple_rnn_50_while_simple_rnn_50_while_loop_counter$simple_rnn_50/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_50/while/IdentityIdentitysimple_rnn_50/while/add_1:z:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: ¢
simple_rnn_50/while/Identity_1Identity:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_50/while/Identity_2Identitysimple_rnn_50/while/add:z:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: °
simple_rnn_50/while/Identity_3IdentityHsimple_rnn_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: ¨
simple_rnn_50/while/Identity_4Identity/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0^simple_rnn_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_50/while/NoOpNoOp>^simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp=^simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp?^simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_50_while_identity%simple_rnn_50/while/Identity:output:0"I
simple_rnn_50_while_identity_1'simple_rnn_50/while/Identity_1:output:0"I
simple_rnn_50_while_identity_2'simple_rnn_50/while/Identity_2:output:0"I
simple_rnn_50_while_identity_3'simple_rnn_50/while/Identity_3:output:0"I
simple_rnn_50_while_identity_4'simple_rnn_50/while/Identity_4:output:0"h
1simple_rnn_50_while_simple_rnn_50_strided_slice_13simple_rnn_50_while_simple_rnn_50_strided_slice_1_0"
Fsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resourceHsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resourceIsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"
Esimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resourceGsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0"à
msimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensorosimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2~
=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2|
<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp2
>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 


Ó
.__inference_sequential_25_layer_call_fn_189911
simple_rnn_50_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:	@à
	unknown_3:	à
	unknown_4:
àà
	unknown_5:	à
	unknown_6:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_189892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input

¼
.__inference_simple_rnn_51_layer_call_fn_191479

inputs
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_190069p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê,
Ù
while_body_190003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
×

´
simple_rnn_51_while_cond_1905988
4simple_rnn_51_while_simple_rnn_51_while_loop_counter>
:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations#
simple_rnn_51_while_placeholder%
!simple_rnn_51_while_placeholder_1%
!simple_rnn_51_while_placeholder_2:
6simple_rnn_51_while_less_simple_rnn_51_strided_slice_1P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190598___redundant_placeholder0P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190598___redundant_placeholder1P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190598___redundant_placeholder2P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190598___redundant_placeholder3 
simple_rnn_51_while_identity

simple_rnn_51/while/LessLesssimple_rnn_51_while_placeholder6simple_rnn_51_while_less_simple_rnn_51_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_51/while/IdentityIdentitysimple_rnn_51/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_51_while_identity%simple_rnn_51/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
Ñ	
É
$__inference_signature_wrapper_190402
simple_rnn_50_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:	@à
	unknown_3:	à
	unknown_4:
àà
	unknown_5:	à
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_189015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input
!
à
while_body_189543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_51_189565_0:	@à0
!while_simple_rnn_cell_51_189567_0:	à5
!while_simple_rnn_cell_51_189569_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_51_189565:	@à.
while_simple_rnn_cell_51_189567:	à3
while_simple_rnn_cell_51_189569:
àà¢0while/simple_rnn_cell_51/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0¬
0while/simple_rnn_cell_51/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_51_189565_0!while_simple_rnn_cell_51_189567_0!while_simple_rnn_cell_51_189569_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189487â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_51/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_51/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà

while/NoOpNoOp1^while/simple_rnn_cell_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_51_189565!while_simple_rnn_cell_51_189565_0"D
while_simple_rnn_cell_51_189567!while_simple_rnn_cell_51_189567_0"D
while_simple_rnn_cell_51_189569!while_simple_rnn_cell_51_189569_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2d
0while/simple_rnn_cell_51/StatefulPartitionedCall0while/simple_rnn_cell_51/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_191005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191005___redundant_placeholder04
0while_while_cond_191005___redundant_placeholder14
0while_while_cond_191005___redundant_placeholder24
0while_while_cond_191005___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
þ
é
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189187

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
ê,
Ù
while_body_189793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 

¼
.__inference_simple_rnn_51_layer_call_fn_191468

inputs
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189859p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü6
 
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189306

inputs+
simple_rnn_cell_50_189231:@'
simple_rnn_cell_50_189233:@+
simple_rnn_cell_50_189235:@@
identity¢*simple_rnn_cell_50/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskï
*simple_rnn_cell_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_50_189231simple_rnn_cell_50_189233simple_rnn_cell_50_189235*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189187n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_50_189231simple_rnn_cell_50_189233simple_rnn_cell_50_189235*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189243*
condR
while_cond_189242*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_50/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2X
*simple_rnn_cell_50/StatefulPartitionedCall*simple_rnn_cell_50/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
¾
.__inference_simple_rnn_51_layer_call_fn_191446
inputs_0
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189443p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0
¡
¾
.__inference_simple_rnn_51_layer_call_fn_191457
inputs_0
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189606p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0
Ü
ª
while_cond_190002
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_190002___redundant_placeholder04
0while_while_cond_190002___redundant_placeholder14
0while_while_cond_190002___redundant_placeholder24
0while_while_cond_190002___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:

¸
.__inference_simple_rnn_50_layer_call_fn_190960

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_190226s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾@
Æ
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189859

inputsD
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189793*
condR
while_cond_189792*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê,
Ù
while_body_191637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
í£
	
I__inference_sequential_25_layer_call_and_return_conditional_losses_190673

inputsQ
?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource:@N
@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource:@S
Asimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@R
?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource:	@àO
@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource:	àU
Asimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà:
'dense_25_matmul_readvariableop_resource:	à6
(dense_25_biasadd_readvariableop_resource:
identity¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp¢8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp¢simple_rnn_50/while¢7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp¢8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp¢simple_rnn_51/whileI
simple_rnn_50/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_50/strided_sliceStridedSlicesimple_rnn_50/Shape:output:0*simple_rnn_50/strided_slice/stack:output:0,simple_rnn_50/strided_slice/stack_1:output:0,simple_rnn_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
simple_rnn_50/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_50/zeros/mulMul$simple_rnn_50/strided_slice:output:0"simple_rnn_50/zeros/mul/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_50/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
simple_rnn_50/zeros/LessLesssimple_rnn_50/zeros/mul:z:0#simple_rnn_50/zeros/Less/y:output:0*
T0*
_output_shapes
: ^
simple_rnn_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_50/zeros/packedPack$simple_rnn_50/strided_slice:output:0%simple_rnn_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_50/zerosFill#simple_rnn_50/zeros/packed:output:0"simple_rnn_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
simple_rnn_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_50/transpose	Transposeinputs%simple_rnn_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
simple_rnn_50/Shape_1Shapesimple_rnn_50/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
simple_rnn_50/strided_slice_1StridedSlicesimple_rnn_50/Shape_1:output:0,simple_rnn_50/strided_slice_1/stack:output:0.simple_rnn_50/strided_slice_1/stack_1:output:0.simple_rnn_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
simple_rnn_50/TensorArrayV2TensorListReserve2simple_rnn_50/TensorArrayV2/element_shape:output:0&simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csimple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5simple_rnn_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_50/transpose:y:0Lsimple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#simple_rnn_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
simple_rnn_50/strided_slice_2StridedSlicesimple_rnn_50/transpose:y:0,simple_rnn_50/strided_slice_2/stack:output:0.simple_rnn_50/strided_slice_2/stack_1:output:0.simple_rnn_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¶
6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ë
'simple_rnn_50/simple_rnn_cell_50/MatMulMatMul&simple_rnn_50/strided_slice_2:output:0>simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@´
7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
(simple_rnn_50/simple_rnn_cell_50/BiasAddBiasAdd1simple_rnn_50/simple_rnn_cell_50/MatMul:product:0?simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Å
)simple_rnn_50/simple_rnn_cell_50/MatMul_1MatMulsimple_rnn_50/zeros:output:0@simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
$simple_rnn_50/simple_rnn_cell_50/addAddV21simple_rnn_50/simple_rnn_cell_50/BiasAdd:output:03simple_rnn_50/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%simple_rnn_50/simple_rnn_cell_50/TanhTanh(simple_rnn_50/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
+simple_rnn_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   â
simple_rnn_50/TensorArrayV2_1TensorListReserve4simple_rnn_50/TensorArrayV2_1/element_shape:output:0&simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
simple_rnn_50/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 simple_rnn_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_50/whileWhile)simple_rnn_50/while/loop_counter:output:0/simple_rnn_50/while/maximum_iterations:output:0simple_rnn_50/time:output:0&simple_rnn_50/TensorArrayV2_1:handle:0simple_rnn_50/zeros:output:0&simple_rnn_50/strided_slice_1:output:0Esimple_rnn_50/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resourceAsimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_50_while_body_190490*+
cond#R!
simple_rnn_50_while_cond_190489*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
>simple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ì
0simple_rnn_50/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_50/while:output:3Gsimple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0v
#simple_rnn_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%simple_rnn_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
simple_rnn_50/strided_slice_3StridedSlice9simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_50/strided_slice_3/stack:output:0.simple_rnn_50/strided_slice_3/stack_1:output:0.simple_rnn_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_masks
simple_rnn_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
simple_rnn_50/transpose_1	Transpose9simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
dropout_28/IdentityIdentitysimple_rnn_50/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
simple_rnn_51/ShapeShapedropout_28/Identity:output:0*
T0*
_output_shapes
:k
!simple_rnn_51/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_51/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_51/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_51/strided_sliceStridedSlicesimple_rnn_51/Shape:output:0*simple_rnn_51/strided_slice/stack:output:0,simple_rnn_51/strided_slice/stack_1:output:0,simple_rnn_51/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
simple_rnn_51/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à
simple_rnn_51/zeros/mulMul$simple_rnn_51/strided_slice:output:0"simple_rnn_51/zeros/mul/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_51/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
simple_rnn_51/zeros/LessLesssimple_rnn_51/zeros/mul:z:0#simple_rnn_51/zeros/Less/y:output:0*
T0*
_output_shapes
: _
simple_rnn_51/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :à
simple_rnn_51/zeros/packedPack$simple_rnn_51/strided_slice:output:0%simple_rnn_51/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_51/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_51/zerosFill#simple_rnn_51/zeros/packed:output:0"simple_rnn_51/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàq
simple_rnn_51/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_51/transpose	Transposedropout_28/Identity:output:0%simple_rnn_51/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
simple_rnn_51/Shape_1Shapesimple_rnn_51/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_51/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_51/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
simple_rnn_51/strided_slice_1StridedSlicesimple_rnn_51/Shape_1:output:0,simple_rnn_51/strided_slice_1/stack:output:0.simple_rnn_51/strided_slice_1/stack_1:output:0.simple_rnn_51/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_51/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
simple_rnn_51/TensorArrayV2TensorListReserve2simple_rnn_51/TensorArrayV2/element_shape:output:0&simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csimple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
5simple_rnn_51/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_51/transpose:y:0Lsimple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#simple_rnn_51/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_51/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
simple_rnn_51/strided_slice_2StridedSlicesimple_rnn_51/transpose:y:0,simple_rnn_51/strided_slice_2/stack:output:0.simple_rnn_51/strided_slice_2/stack_1:output:0.simple_rnn_51/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask·
6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0Ì
'simple_rnn_51/simple_rnn_cell_51/MatMulMatMul&simple_rnn_51/strided_slice_2:output:0>simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàµ
7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ú
(simple_rnn_51/simple_rnn_cell_51/BiasAddBiasAdd1simple_rnn_51/simple_rnn_cell_51/MatMul:product:0?simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà¼
8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0Æ
)simple_rnn_51/simple_rnn_cell_51/MatMul_1MatMulsimple_rnn_51/zeros:output:0@simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÈ
$simple_rnn_51/simple_rnn_cell_51/addAddV21simple_rnn_51/simple_rnn_cell_51/BiasAdd:output:03simple_rnn_51/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
%simple_rnn_51/simple_rnn_cell_51/TanhTanh(simple_rnn_51/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà|
+simple_rnn_51/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   â
simple_rnn_51/TensorArrayV2_1TensorListReserve4simple_rnn_51/TensorArrayV2_1/element_shape:output:0&simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
simple_rnn_51/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_51/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 simple_rnn_51/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_51/whileWhile)simple_rnn_51/while/loop_counter:output:0/simple_rnn_51/while/maximum_iterations:output:0simple_rnn_51/time:output:0&simple_rnn_51/TensorArrayV2_1:handle:0simple_rnn_51/zeros:output:0&simple_rnn_51/strided_slice_1:output:0Esimple_rnn_51/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resourceAsimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_51_while_body_190599*+
cond#R!
simple_rnn_51_while_cond_190598*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
>simple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   í
0simple_rnn_51/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_51/while:output:3Gsimple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0v
#simple_rnn_51/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%simple_rnn_51/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
simple_rnn_51/strided_slice_3StridedSlice9simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_51/strided_slice_3/stack:output:0.simple_rnn_51/strided_slice_3/stack_1:output:0.simple_rnn_51/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_masks
simple_rnn_51/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Á
simple_rnn_51/transpose_1	Transpose9simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_51/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
dropout_29/IdentityIdentity&simple_rnn_51/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	à*
dtype0
dense_25/MatMulMatMuldropout_29/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_25/SoftmaxSoftmaxdense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_25/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp8^simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp7^simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp9^simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp^simple_rnn_50/while8^simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp7^simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp9^simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp^simple_rnn_51/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2r
7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp2p
6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp2t
8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp2*
simple_rnn_50/whilesimple_rnn_50/while2r
7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp2p
6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp2t
8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp2*
simple_rnn_51/whilesimple_rnn_51/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

ß
3__inference_simple_rnn_cell_51_layer_call_fn_192064

inputs
states_0
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189487p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
states/0
µ
º
.__inference_simple_rnn_50_layer_call_fn_190938
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189306|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ú
ª
while_cond_190159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_190159___redundant_placeholder04
0while_while_cond_190159___redundant_placeholder14
0while_while_cond_190159___redundant_placeholder24
0while_while_cond_190159___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
²³
	
I__inference_sequential_25_layer_call_and_return_conditional_losses_190916

inputsQ
?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource:@N
@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource:@S
Asimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@R
?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource:	@àO
@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource:	àU
Asimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà:
'dense_25_matmul_readvariableop_resource:	à6
(dense_25_biasadd_readvariableop_resource:
identity¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp¢8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp¢simple_rnn_50/while¢7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp¢8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp¢simple_rnn_51/whileI
simple_rnn_50/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_50/strided_sliceStridedSlicesimple_rnn_50/Shape:output:0*simple_rnn_50/strided_slice/stack:output:0,simple_rnn_50/strided_slice/stack_1:output:0,simple_rnn_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
simple_rnn_50/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_50/zeros/mulMul$simple_rnn_50/strided_slice:output:0"simple_rnn_50/zeros/mul/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_50/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
simple_rnn_50/zeros/LessLesssimple_rnn_50/zeros/mul:z:0#simple_rnn_50/zeros/Less/y:output:0*
T0*
_output_shapes
: ^
simple_rnn_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_50/zeros/packedPack$simple_rnn_50/strided_slice:output:0%simple_rnn_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_50/zerosFill#simple_rnn_50/zeros/packed:output:0"simple_rnn_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
simple_rnn_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_50/transpose	Transposeinputs%simple_rnn_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
simple_rnn_50/Shape_1Shapesimple_rnn_50/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
simple_rnn_50/strided_slice_1StridedSlicesimple_rnn_50/Shape_1:output:0,simple_rnn_50/strided_slice_1/stack:output:0.simple_rnn_50/strided_slice_1/stack_1:output:0.simple_rnn_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
simple_rnn_50/TensorArrayV2TensorListReserve2simple_rnn_50/TensorArrayV2/element_shape:output:0&simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csimple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5simple_rnn_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_50/transpose:y:0Lsimple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#simple_rnn_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
simple_rnn_50/strided_slice_2StridedSlicesimple_rnn_50/transpose:y:0,simple_rnn_50/strided_slice_2/stack:output:0.simple_rnn_50/strided_slice_2/stack_1:output:0.simple_rnn_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¶
6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ë
'simple_rnn_50/simple_rnn_cell_50/MatMulMatMul&simple_rnn_50/strided_slice_2:output:0>simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@´
7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
(simple_rnn_50/simple_rnn_cell_50/BiasAddBiasAdd1simple_rnn_50/simple_rnn_cell_50/MatMul:product:0?simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Å
)simple_rnn_50/simple_rnn_cell_50/MatMul_1MatMulsimple_rnn_50/zeros:output:0@simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
$simple_rnn_50/simple_rnn_cell_50/addAddV21simple_rnn_50/simple_rnn_cell_50/BiasAdd:output:03simple_rnn_50/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%simple_rnn_50/simple_rnn_cell_50/TanhTanh(simple_rnn_50/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
+simple_rnn_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   â
simple_rnn_50/TensorArrayV2_1TensorListReserve4simple_rnn_50/TensorArrayV2_1/element_shape:output:0&simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
simple_rnn_50/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 simple_rnn_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_50/whileWhile)simple_rnn_50/while/loop_counter:output:0/simple_rnn_50/while/maximum_iterations:output:0simple_rnn_50/time:output:0&simple_rnn_50/TensorArrayV2_1:handle:0simple_rnn_50/zeros:output:0&simple_rnn_50/strided_slice_1:output:0Esimple_rnn_50/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource@simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resourceAsimple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_50_while_body_190719*+
cond#R!
simple_rnn_50_while_cond_190718*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
>simple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ì
0simple_rnn_50/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_50/while:output:3Gsimple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0v
#simple_rnn_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%simple_rnn_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
simple_rnn_50/strided_slice_3StridedSlice9simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_50/strided_slice_3/stack:output:0.simple_rnn_50/strided_slice_3/stack_1:output:0.simple_rnn_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_masks
simple_rnn_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
simple_rnn_50/transpose_1	Transpose9simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?
dropout_28/dropout/MulMulsimple_rnn_50/transpose_1:y:0!dropout_28/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
dropout_28/dropout/ShapeShapesimple_rnn_50/transpose_1:y:0*
T0*
_output_shapes
:¦
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=Ë
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
simple_rnn_51/ShapeShapedropout_28/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!simple_rnn_51/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_51/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_51/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_51/strided_sliceStridedSlicesimple_rnn_51/Shape:output:0*simple_rnn_51/strided_slice/stack:output:0,simple_rnn_51/strided_slice/stack_1:output:0,simple_rnn_51/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
simple_rnn_51/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à
simple_rnn_51/zeros/mulMul$simple_rnn_51/strided_slice:output:0"simple_rnn_51/zeros/mul/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_51/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
simple_rnn_51/zeros/LessLesssimple_rnn_51/zeros/mul:z:0#simple_rnn_51/zeros/Less/y:output:0*
T0*
_output_shapes
: _
simple_rnn_51/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :à
simple_rnn_51/zeros/packedPack$simple_rnn_51/strided_slice:output:0%simple_rnn_51/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_51/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_51/zerosFill#simple_rnn_51/zeros/packed:output:0"simple_rnn_51/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàq
simple_rnn_51/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_51/transpose	Transposedropout_28/dropout/Mul_1:z:0%simple_rnn_51/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
simple_rnn_51/Shape_1Shapesimple_rnn_51/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_51/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_51/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
simple_rnn_51/strided_slice_1StridedSlicesimple_rnn_51/Shape_1:output:0,simple_rnn_51/strided_slice_1/stack:output:0.simple_rnn_51/strided_slice_1/stack_1:output:0.simple_rnn_51/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_51/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
simple_rnn_51/TensorArrayV2TensorListReserve2simple_rnn_51/TensorArrayV2/element_shape:output:0&simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Csimple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
5simple_rnn_51/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_51/transpose:y:0Lsimple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#simple_rnn_51/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_51/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
simple_rnn_51/strided_slice_2StridedSlicesimple_rnn_51/transpose:y:0,simple_rnn_51/strided_slice_2/stack:output:0.simple_rnn_51/strided_slice_2/stack_1:output:0.simple_rnn_51/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask·
6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0Ì
'simple_rnn_51/simple_rnn_cell_51/MatMulMatMul&simple_rnn_51/strided_slice_2:output:0>simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàµ
7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0Ú
(simple_rnn_51/simple_rnn_cell_51/BiasAddBiasAdd1simple_rnn_51/simple_rnn_cell_51/MatMul:product:0?simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà¼
8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0Æ
)simple_rnn_51/simple_rnn_cell_51/MatMul_1MatMulsimple_rnn_51/zeros:output:0@simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÈ
$simple_rnn_51/simple_rnn_cell_51/addAddV21simple_rnn_51/simple_rnn_cell_51/BiasAdd:output:03simple_rnn_51/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
%simple_rnn_51/simple_rnn_cell_51/TanhTanh(simple_rnn_51/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà|
+simple_rnn_51/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   â
simple_rnn_51/TensorArrayV2_1TensorListReserve4simple_rnn_51/TensorArrayV2_1/element_shape:output:0&simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
simple_rnn_51/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_51/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 simple_rnn_51/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_51/whileWhile)simple_rnn_51/while/loop_counter:output:0/simple_rnn_51/while/maximum_iterations:output:0simple_rnn_51/time:output:0&simple_rnn_51/TensorArrayV2_1:handle:0simple_rnn_51/zeros:output:0&simple_rnn_51/strided_slice_1:output:0Esimple_rnn_51/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource@simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resourceAsimple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_51_while_body_190835*+
cond#R!
simple_rnn_51_while_cond_190834*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
>simple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   í
0simple_rnn_51/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_51/while:output:3Gsimple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0v
#simple_rnn_51/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%simple_rnn_51/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_51/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
simple_rnn_51/strided_slice_3StridedSlice9simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_51/strided_slice_3/stack:output:0.simple_rnn_51/strided_slice_3/stack_1:output:0.simple_rnn_51/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_masks
simple_rnn_51/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Á
simple_rnn_51/transpose_1	Transpose9simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_51/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà]
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?
dropout_29/dropout/MulMul&simple_rnn_51/strided_slice_3:output:0!dropout_29/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
dropout_29/dropout/ShapeShape&simple_rnn_51/strided_slice_3:output:0*
T0*
_output_shapes
:£
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=È
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	à*
dtype0
dense_25/MatMulMatMuldropout_29/dropout/Mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_25/SoftmaxSoftmaxdense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_25/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp8^simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp7^simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp9^simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp^simple_rnn_50/while8^simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp7^simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp9^simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp^simple_rnn_51/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2r
7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp7simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp2p
6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp6simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp2t
8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp8simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp2*
simple_rnn_50/whilesimple_rnn_50/while2r
7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp7simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp2p
6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp6simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp2t
8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp8simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp2*
simple_rnn_51/whilesimple_rnn_51/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

ö
D__inference_dense_25_layer_call_and_return_conditional_losses_189885

inputs1
matmul_readvariableop_resource:	à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	à*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ö,
Ñ
while_body_191006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ö,
Ñ
while_body_191342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
×

´
simple_rnn_51_while_cond_1908348
4simple_rnn_51_while_simple_rnn_51_while_loop_counter>
:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations#
simple_rnn_51_while_placeholder%
!simple_rnn_51_while_placeholder_1%
!simple_rnn_51_while_placeholder_2:
6simple_rnn_51_while_less_simple_rnn_51_strided_slice_1P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190834___redundant_placeholder0P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190834___redundant_placeholder1P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190834___redundant_placeholder2P
Lsimple_rnn_51_while_simple_rnn_51_while_cond_190834___redundant_placeholder3 
simple_rnn_51_while_identity

simple_rnn_51/while/LessLesssimple_rnn_51_while_placeholder6simple_rnn_51_while_less_simple_rnn_51_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_51/while/IdentityIdentitysimple_rnn_51/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_51_while_identity%simple_rnn_51/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
£@
Â
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191408

inputsC
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191342*
condR
while_cond_191341*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
G
+__inference_dropout_28_layer_call_fn_191413

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_189746d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_29_layer_call_and_return_conditional_losses_189941

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ö,
Ñ
while_body_191230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ö,
Ñ
while_body_189667
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ã9
õ
simple_rnn_50_while_body_1904908
4simple_rnn_50_while_simple_rnn_50_while_loop_counter>
:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations#
simple_rnn_50_while_placeholder%
!simple_rnn_50_while_placeholder_1%
!simple_rnn_50_while_placeholder_27
3simple_rnn_50_while_simple_rnn_50_strided_slice_1_0s
osimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@V
Hsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@[
Isimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@ 
simple_rnn_50_while_identity"
simple_rnn_50_while_identity_1"
simple_rnn_50_while_identity_2"
simple_rnn_50_while_identity_3"
simple_rnn_50_while_identity_45
1simple_rnn_50_while_simple_rnn_50_strided_slice_1q
msimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource:@T
Fsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource:@Y
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp¢>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
Esimple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7simple_rnn_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_50_while_placeholderNsimple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ä
<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0ï
-simple_rnn_50/while/simple_rnn_cell_50/MatMulMatMul>simple_rnn_50/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0ë
.simple_rnn_50/while/simple_rnn_cell_50/BiasAddBiasAdd7simple_rnn_50/while/simple_rnn_cell_50/MatMul:product:0Esimple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Ö
/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1MatMul!simple_rnn_50_while_placeholder_2Fsimple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ù
*simple_rnn_50/while/simple_rnn_cell_50/addAddV27simple_rnn_50/while/simple_rnn_cell_50/BiasAdd:output:09simple_rnn_50/while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+simple_rnn_50/while/simple_rnn_cell_50/TanhTanh.simple_rnn_50/while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
8simple_rnn_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_50_while_placeholder_1simple_rnn_50_while_placeholder/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ[
simple_rnn_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_50/while/addAddV2simple_rnn_50_while_placeholder"simple_rnn_50/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_50/while/add_1AddV24simple_rnn_50_while_simple_rnn_50_while_loop_counter$simple_rnn_50/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_50/while/IdentityIdentitysimple_rnn_50/while/add_1:z:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: ¢
simple_rnn_50/while/Identity_1Identity:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_50/while/Identity_2Identitysimple_rnn_50/while/add:z:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: °
simple_rnn_50/while/Identity_3IdentityHsimple_rnn_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_50/while/NoOp*
T0*
_output_shapes
: ¨
simple_rnn_50/while/Identity_4Identity/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0^simple_rnn_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_50/while/NoOpNoOp>^simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp=^simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp?^simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_50_while_identity%simple_rnn_50/while/Identity:output:0"I
simple_rnn_50_while_identity_1'simple_rnn_50/while/Identity_1:output:0"I
simple_rnn_50_while_identity_2'simple_rnn_50/while/Identity_2:output:0"I
simple_rnn_50_while_identity_3'simple_rnn_50/while/Identity_3:output:0"I
simple_rnn_50_while_identity_4'simple_rnn_50/while/Identity_4:output:0"h
1simple_rnn_50_while_simple_rnn_50_strided_slice_13simple_rnn_50_while_simple_rnn_50_strided_slice_1_0"
Fsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resourceHsimple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"
Gsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resourceIsimple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"
Esimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resourceGsimple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0"à
msimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensorosimple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2~
=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp=simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2|
<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp<simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp2
>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp>simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ü
ª
while_cond_191748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191748___redundant_placeholder04
0while_while_cond_191748___redundant_placeholder14
0while_while_cond_191748___redundant_placeholder24
0while_while_cond_191748___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
ê,
Ù
while_body_191749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
Ò
Ô
I__inference_sequential_25_layer_call_and_return_conditional_losses_190283

inputs&
simple_rnn_50_190261:@"
simple_rnn_50_190263:@&
simple_rnn_50_190265:@@'
simple_rnn_51_190269:	@à#
simple_rnn_51_190271:	à(
simple_rnn_51_190273:
àà"
dense_25_190277:	à
dense_25_190279:
identity¢ dense_25/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢%simple_rnn_50/StatefulPartitionedCall¢%simple_rnn_51/StatefulPartitionedCall£
%simple_rnn_50/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_50_190261simple_rnn_50_190263simple_rnn_50_190265*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_190226û
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_190098Å
%simple_rnn_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0simple_rnn_51_190269simple_rnn_51_190271simple_rnn_51_190273*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_190069
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_51/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189941
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0dense_25_190277dense_25_190279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_189885x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_25/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall&^simple_rnn_50/StatefulPartitionedCall&^simple_rnn_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2N
%simple_rnn_50/StatefulPartitionedCall%simple_rnn_50/StatefulPartitionedCall2N
%simple_rnn_51/StatefulPartitionedCall%simple_rnn_51/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
º
.__inference_simple_rnn_50_layer_call_fn_190927
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189143|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
£@
Â
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_190226

inputsC
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_190160*
condR
while_cond_190159*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_29_layer_call_and_return_conditional_losses_191954

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
È

ß
3__inference_simple_rnn_cell_51_layer_call_fn_192050

inputs
states_0
unknown:	@à
	unknown_0:	à
	unknown_1:
àà
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189367p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
states/0
é
d
F__inference_dropout_28_layer_call_and_return_conditional_losses_189746

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ

´
simple_rnn_50_while_cond_1907188
4simple_rnn_50_while_simple_rnn_50_while_loop_counter>
:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations#
simple_rnn_50_while_placeholder%
!simple_rnn_50_while_placeholder_1%
!simple_rnn_50_while_placeholder_2:
6simple_rnn_50_while_less_simple_rnn_50_strided_slice_1P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190718___redundant_placeholder0P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190718___redundant_placeholder1P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190718___redundant_placeholder2P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190718___redundant_placeholder3 
simple_rnn_50_while_identity

simple_rnn_50/while/LessLesssimple_rnn_50_while_placeholder6simple_rnn_50_while_less_simple_rnn_50_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_50/while/IdentityIdentitysimple_rnn_50/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_50_while_identity%simple_rnn_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¾

Û
3__inference_simple_rnn_cell_50_layer_call_fn_192002

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189187o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
Õ

I__inference_sequential_25_layer_call_and_return_conditional_losses_189892

inputs&
simple_rnn_50_189734:@"
simple_rnn_50_189736:@&
simple_rnn_50_189738:@@'
simple_rnn_51_189860:	@à#
simple_rnn_51_189862:	à(
simple_rnn_51_189864:
àà"
dense_25_189886:	à
dense_25_189888:
identity¢ dense_25/StatefulPartitionedCall¢%simple_rnn_50/StatefulPartitionedCall¢%simple_rnn_51/StatefulPartitionedCall£
%simple_rnn_50/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_50_189734simple_rnn_50_189736simple_rnn_50_189738*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189733ë
dropout_28/PartitionedCallPartitionedCall.simple_rnn_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_189746½
%simple_rnn_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0simple_rnn_51_189860simple_rnn_51_189862simple_rnn_51_189864*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189859è
dropout_29/PartitionedCallPartitionedCall.simple_rnn_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189872
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0dense_25_189886dense_25_189888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_189885x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
NoOpNoOp!^dense_25/StatefulPartitionedCall&^simple_rnn_50/StatefulPartitionedCall&^simple_rnn_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2N
%simple_rnn_50/StatefulPartitionedCall%simple_rnn_50/StatefulPartitionedCall2N
%simple_rnn_51/StatefulPartitionedCall%simple_rnn_51/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
ª
while_cond_191117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191117___redundant_placeholder04
0while_while_cond_191117___redundant_placeholder14
0while_while_cond_191117___redundant_placeholder24
0while_while_cond_191117___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
á@
Ä
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191072
inputs_0C
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191006*
condR
while_cond_191005*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
é
d
F__inference_dropout_28_layer_call_and_return_conditional_losses_191423

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ïF

-sequential_25_simple_rnn_50_while_body_188832T
Psequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_loop_counterZ
Vsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_maximum_iterations1
-sequential_25_simple_rnn_50_while_placeholder3
/sequential_25_simple_rnn_50_while_placeholder_13
/sequential_25_simple_rnn_50_while_placeholder_2S
Osequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_strided_slice_1_0
sequential_25_simple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@d
Vsequential_25_simple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@i
Wsequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@.
*sequential_25_simple_rnn_50_while_identity0
,sequential_25_simple_rnn_50_while_identity_10
,sequential_25_simple_rnn_50_while_identity_20
,sequential_25_simple_rnn_50_while_identity_30
,sequential_25_simple_rnn_50_while_identity_4Q
Msequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_strided_slice_1
sequential_25_simple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_50_tensorarrayunstack_tensorlistfromtensore
Ssequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource:@b
Tsequential_25_simple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource:@g
Usequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢Ksequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢Jsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp¢Lsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp¤
Ssequential_25/simple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ³
Esequential_25/simple_rnn_50/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_25_simple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0-sequential_25_simple_rnn_50_while_placeholder\sequential_25/simple_rnn_50/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0à
Jsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOpUsequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0
;sequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMulMatMulLsequential_25/simple_rnn_50/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
Ksequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOpVsequential_25_simple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
<sequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAddBiasAddEsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul:product:0Ssequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ä
Lsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpWsequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0
=sequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1MatMul/sequential_25_simple_rnn_50_while_placeholder_2Tsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
8sequential_25/simple_rnn_50/while/simple_rnn_cell_50/addAddV2Esequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd:output:0Gsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@±
9sequential_25/simple_rnn_50/while/simple_rnn_cell_50/TanhTanh<sequential_25/simple_rnn_50/while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
Fsequential_25/simple_rnn_50/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_25_simple_rnn_50_while_placeholder_1-sequential_25_simple_rnn_50_while_placeholder=sequential_25/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒi
'sequential_25/simple_rnn_50/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
%sequential_25/simple_rnn_50/while/addAddV2-sequential_25_simple_rnn_50_while_placeholder0sequential_25/simple_rnn_50/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_25/simple_rnn_50/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :×
'sequential_25/simple_rnn_50/while/add_1AddV2Psequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_loop_counter2sequential_25/simple_rnn_50/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*sequential_25/simple_rnn_50/while/IdentityIdentity+sequential_25/simple_rnn_50/while/add_1:z:0'^sequential_25/simple_rnn_50/while/NoOp*
T0*
_output_shapes
: Ú
,sequential_25/simple_rnn_50/while/Identity_1IdentityVsequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_while_maximum_iterations'^sequential_25/simple_rnn_50/while/NoOp*
T0*
_output_shapes
: ­
,sequential_25/simple_rnn_50/while/Identity_2Identity)sequential_25/simple_rnn_50/while/add:z:0'^sequential_25/simple_rnn_50/while/NoOp*
T0*
_output_shapes
: Ú
,sequential_25/simple_rnn_50/while/Identity_3IdentityVsequential_25/simple_rnn_50/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_25/simple_rnn_50/while/NoOp*
T0*
_output_shapes
: Ò
,sequential_25/simple_rnn_50/while/Identity_4Identity=sequential_25/simple_rnn_50/while/simple_rnn_cell_50/Tanh:y:0'^sequential_25/simple_rnn_50/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
&sequential_25/simple_rnn_50/while/NoOpNoOpL^sequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpK^sequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOpM^sequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_25_simple_rnn_50_while_identity3sequential_25/simple_rnn_50/while/Identity:output:0"e
,sequential_25_simple_rnn_50_while_identity_15sequential_25/simple_rnn_50/while/Identity_1:output:0"e
,sequential_25_simple_rnn_50_while_identity_25sequential_25/simple_rnn_50/while/Identity_2:output:0"e
,sequential_25_simple_rnn_50_while_identity_35sequential_25/simple_rnn_50/while/Identity_3:output:0"e
,sequential_25_simple_rnn_50_while_identity_45sequential_25/simple_rnn_50/while/Identity_4:output:0" 
Msequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_strided_slice_1Osequential_25_simple_rnn_50_while_sequential_25_simple_rnn_50_strided_slice_1_0"®
Tsequential_25_simple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resourceVsequential_25_simple_rnn_50_while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"°
Usequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resourceWsequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"¬
Ssequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resourceUsequential_25_simple_rnn_50_while_simple_rnn_cell_50_matmul_readvariableop_resource_0"
sequential_25_simple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_50_tensorarrayunstack_tensorlistfromtensorsequential_25_simple_rnn_50_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_50_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2
Ksequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpKsequential_25/simple_rnn_50/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2
Jsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOpJsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul/ReadVariableOp2
Lsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOpLsequential_25/simple_rnn_50/while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
!
à
while_body_189380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_51_189402_0:	@à0
!while_simple_rnn_cell_51_189404_0:	à5
!while_simple_rnn_cell_51_189406_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_51_189402:	@à.
while_simple_rnn_cell_51_189404:	à3
while_simple_rnn_cell_51_189406:
àà¢0while/simple_rnn_cell_51/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0¬
0while/simple_rnn_cell_51/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_51_189402_0!while_simple_rnn_cell_51_189404_0!while_simple_rnn_cell_51_189406_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189367â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_51/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_51/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà

while/NoOpNoOp1^while/simple_rnn_cell_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_51_189402!while_simple_rnn_cell_51_189402_0"D
while_simple_rnn_cell_51_189404!while_simple_rnn_cell_51_189404_0"D
while_simple_rnn_cell_51_189406!while_simple_rnn_cell_51_189406_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2d
0while/simple_rnn_cell_51/StatefulPartitionedCall0while/simple_rnn_cell_51/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
Ü
ª
while_cond_191636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191636___redundant_placeholder04
0while_while_cond_191636___redundant_placeholder14
0while_while_cond_191636___redundant_placeholder24
0while_while_cond_191636___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
¤

ö
D__inference_dense_25_layer_call_and_return_conditional_losses_191974

inputs1
matmul_readvariableop_resource:	à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	à*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ü
ª
while_cond_191860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191860___redundant_placeholder04
0while_while_cond_191860___redundant_placeholder14
0while_while_cond_191860___redundant_placeholder24
0while_while_cond_191860___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
G
¥
-sequential_25_simple_rnn_51_while_body_188941T
Psequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_loop_counterZ
Vsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_maximum_iterations1
-sequential_25_simple_rnn_51_while_placeholder3
/sequential_25_simple_rnn_51_while_placeholder_13
/sequential_25_simple_rnn_51_while_placeholder_2S
Osequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_strided_slice_1_0
sequential_25_simple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0h
Usequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àe
Vsequential_25_simple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àk
Wsequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà.
*sequential_25_simple_rnn_51_while_identity0
,sequential_25_simple_rnn_51_while_identity_10
,sequential_25_simple_rnn_51_while_identity_20
,sequential_25_simple_rnn_51_while_identity_30
,sequential_25_simple_rnn_51_while_identity_4Q
Msequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_strided_slice_1
sequential_25_simple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorf
Ssequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àc
Tsequential_25_simple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource:	ài
Usequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢Ksequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢Jsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp¢Lsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp¤
Ssequential_25/simple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ³
Esequential_25/simple_rnn_51/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_25_simple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0-sequential_25_simple_rnn_51_while_placeholder\sequential_25/simple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0á
Jsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOpUsequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0
;sequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMulMatMulLsequential_25/simple_rnn_51/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàß
Ksequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOpVsequential_25_simple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0
<sequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAddBiasAddEsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul:product:0Ssequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàæ
Lsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpWsequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0
=sequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1MatMul/sequential_25_simple_rnn_51_while_placeholder_2Tsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
8sequential_25/simple_rnn_51/while/simple_rnn_cell_51/addAddV2Esequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd:output:0Gsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà²
9sequential_25/simple_rnn_51/while/simple_rnn_cell_51/TanhTanh<sequential_25/simple_rnn_51/while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàº
Fsequential_25/simple_rnn_51/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_25_simple_rnn_51_while_placeholder_1-sequential_25_simple_rnn_51_while_placeholder=sequential_25/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒi
'sequential_25/simple_rnn_51/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
%sequential_25/simple_rnn_51/while/addAddV2-sequential_25_simple_rnn_51_while_placeholder0sequential_25/simple_rnn_51/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_25/simple_rnn_51/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :×
'sequential_25/simple_rnn_51/while/add_1AddV2Psequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_loop_counter2sequential_25/simple_rnn_51/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*sequential_25/simple_rnn_51/while/IdentityIdentity+sequential_25/simple_rnn_51/while/add_1:z:0'^sequential_25/simple_rnn_51/while/NoOp*
T0*
_output_shapes
: Ú
,sequential_25/simple_rnn_51/while/Identity_1IdentityVsequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_while_maximum_iterations'^sequential_25/simple_rnn_51/while/NoOp*
T0*
_output_shapes
: ­
,sequential_25/simple_rnn_51/while/Identity_2Identity)sequential_25/simple_rnn_51/while/add:z:0'^sequential_25/simple_rnn_51/while/NoOp*
T0*
_output_shapes
: Ú
,sequential_25/simple_rnn_51/while/Identity_3IdentityVsequential_25/simple_rnn_51/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_25/simple_rnn_51/while/NoOp*
T0*
_output_shapes
: Ó
,sequential_25/simple_rnn_51/while/Identity_4Identity=sequential_25/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0'^sequential_25/simple_rnn_51/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
&sequential_25/simple_rnn_51/while/NoOpNoOpL^sequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpK^sequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOpM^sequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_25_simple_rnn_51_while_identity3sequential_25/simple_rnn_51/while/Identity:output:0"e
,sequential_25_simple_rnn_51_while_identity_15sequential_25/simple_rnn_51/while/Identity_1:output:0"e
,sequential_25_simple_rnn_51_while_identity_25sequential_25/simple_rnn_51/while/Identity_2:output:0"e
,sequential_25_simple_rnn_51_while_identity_35sequential_25/simple_rnn_51/while/Identity_3:output:0"e
,sequential_25_simple_rnn_51_while_identity_45sequential_25/simple_rnn_51/while/Identity_4:output:0" 
Msequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_strided_slice_1Osequential_25_simple_rnn_51_while_sequential_25_simple_rnn_51_strided_slice_1_0"®
Tsequential_25_simple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resourceVsequential_25_simple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"°
Usequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resourceWsequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"¬
Ssequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resourceUsequential_25_simple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0"
sequential_25_simple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorsequential_25_simple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_sequential_25_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2
Ksequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpKsequential_25/simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2
Jsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOpJsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp2
Lsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOpLsequential_25/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
¾@
Æ
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191815

inputsD
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191749*
condR
while_cond_191748*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_29_layer_call_and_return_conditional_losses_189872

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs

ï
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192081

inputs
states_01
matmul_readvariableop_resource:	@à.
biasadd_readvariableop_resource:	à4
 matmul_1_readvariableop_resource:
àà
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàH
TanhTanhadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
states/0
¨
G
+__inference_dropout_29_layer_call_fn_191932

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189872a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
£@
Â
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191296

inputsC
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191230*
condR
while_cond_191229*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


e
F__inference_dropout_28_layer_call_and_return_conditional_losses_190098

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
é
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189067

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
!
Ø
while_body_189080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_50_189102_0:@/
!while_simple_rnn_cell_50_189104_0:@3
!while_simple_rnn_cell_50_189106_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_50_189102:@-
while_simple_rnn_cell_50_189104:@1
while_simple_rnn_cell_50_189106:@@¢0while/simple_rnn_cell_50/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
0while/simple_rnn_cell_50/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_50_189102_0!while_simple_rnn_cell_50_189104_0!while_simple_rnn_cell_50_189106_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189067â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_50/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity9while/simple_rnn_cell_50/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

while/NoOpNoOp1^while/simple_rnn_cell_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_50_189102!while_simple_rnn_cell_50_189102_0"D
while_simple_rnn_cell_50_189104!while_simple_rnn_cell_50_189104_0"D
while_simple_rnn_cell_50_189106!while_simple_rnn_cell_50_189106_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2d
0while/simple_rnn_cell_50/StatefulPartitionedCall0while/simple_rnn_cell_50/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

í
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189487

inputs

states1
matmul_readvariableop_resource:	@à.
biasadd_readvariableop_resource:	à4
 matmul_1_readvariableop_resource:
àà
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàH
TanhTanhadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿà: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_namestates
Ü
ª
while_cond_189542
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189542___redundant_placeholder04
0while_while_cond_189542___redundant_placeholder14
0while_while_cond_189542___redundant_placeholder24
0while_while_cond_189542___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
Ö,
Ñ
while_body_191118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ü6
 
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189143

inputs+
simple_rnn_cell_50_189068:@'
simple_rnn_cell_50_189070:@+
simple_rnn_cell_50_189072:@@
identity¢*simple_rnn_cell_50/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskï
*simple_rnn_cell_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_50_189068simple_rnn_cell_50_189070simple_rnn_cell_50_189072*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189067n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_50_189068simple_rnn_cell_50_189070simple_rnn_cell_50_189072*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189080*
condR
while_cond_189079*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
NoOpNoOp+^simple_rnn_cell_50/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2X
*simple_rnn_cell_50/StatefulPartitionedCall*simple_rnn_cell_50/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_189792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189792___redundant_placeholder04
0while_while_cond_189792___redundant_placeholder14
0while_while_cond_189792___redundant_placeholder24
0while_while_cond_189792___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
Ü	
Æ
.__inference_sequential_25_layer_call_fn_190444

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:	@à
	unknown_3:	à
	unknown_4:
àà
	unknown_5:	à
	unknown_6:
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_190283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_189379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_189379___redundant_placeholder04
0while_while_cond_189379___redundant_placeholder14
0while_while_cond_189379___redundant_placeholder24
0while_while_cond_189379___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:


Ó
.__inference_sequential_25_layer_call_fn_190323
simple_rnn_50_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:	@à
	unknown_3:	à
	unknown_4:
àà
	unknown_5:	à
	unknown_6:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_190283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input
÷9
ý
simple_rnn_51_while_body_1905998
4simple_rnn_51_while_simple_rnn_51_while_loop_counter>
:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations#
simple_rnn_51_while_placeholder%
!simple_rnn_51_while_placeholder_1%
!simple_rnn_51_while_placeholder_27
3simple_rnn_51_while_simple_rnn_51_strided_slice_1_0s
osimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àW
Hsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	à]
Isimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà 
simple_rnn_51_while_identity"
simple_rnn_51_while_identity_1"
simple_rnn_51_while_identity_2"
simple_rnn_51_while_identity_3"
simple_rnn_51_while_identity_45
1simple_rnn_51_while_simple_rnn_51_strided_slice_1q
msimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àU
Fsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource:	à[
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp¢>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
Esimple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ì
7simple_rnn_51/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_51_while_placeholderNsimple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0Å
<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0ð
-simple_rnn_51/while/simple_rnn_cell_51/MatMulMatMul>simple_rnn_51/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÃ
=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0ì
.simple_rnn_51/while/simple_rnn_cell_51/BiasAddBiasAdd7simple_rnn_51/while/simple_rnn_cell_51/MatMul:product:0Esimple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0×
/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1MatMul!simple_rnn_51_while_placeholder_2Fsimple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÚ
*simple_rnn_51/while/simple_rnn_cell_51/addAddV27simple_rnn_51/while/simple_rnn_cell_51/BiasAdd:output:09simple_rnn_51/while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
+simple_rnn_51/while/simple_rnn_cell_51/TanhTanh.simple_rnn_51/while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
8simple_rnn_51/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_51_while_placeholder_1simple_rnn_51_while_placeholder/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ[
simple_rnn_51/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_51/while/addAddV2simple_rnn_51_while_placeholder"simple_rnn_51/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_51/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_51/while/add_1AddV24simple_rnn_51_while_simple_rnn_51_while_loop_counter$simple_rnn_51/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_51/while/IdentityIdentitysimple_rnn_51/while/add_1:z:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: ¢
simple_rnn_51/while/Identity_1Identity:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_51/while/Identity_2Identitysimple_rnn_51/while/add:z:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: °
simple_rnn_51/while/Identity_3IdentityHsimple_rnn_51/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: ©
simple_rnn_51/while/Identity_4Identity/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0^simple_rnn_51/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_51/while/NoOpNoOp>^simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp=^simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp?^simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_51_while_identity%simple_rnn_51/while/Identity:output:0"I
simple_rnn_51_while_identity_1'simple_rnn_51/while/Identity_1:output:0"I
simple_rnn_51_while_identity_2'simple_rnn_51/while/Identity_2:output:0"I
simple_rnn_51_while_identity_3'simple_rnn_51/while/Identity_3:output:0"I
simple_rnn_51_while_identity_4'simple_rnn_51/while/Identity_4:output:0"h
1simple_rnn_51_while_simple_rnn_51_strided_slice_13simple_rnn_51_while_simple_rnn_51_strided_slice_1_0"
Fsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resourceHsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resourceIsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"
Esimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resourceGsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0"à
msimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorosimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2~
=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2|
<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp2
>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
Õ

´
simple_rnn_50_while_cond_1904898
4simple_rnn_50_while_simple_rnn_50_while_loop_counter>
:simple_rnn_50_while_simple_rnn_50_while_maximum_iterations#
simple_rnn_50_while_placeholder%
!simple_rnn_50_while_placeholder_1%
!simple_rnn_50_while_placeholder_2:
6simple_rnn_50_while_less_simple_rnn_50_strided_slice_1P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190489___redundant_placeholder0P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190489___redundant_placeholder1P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190489___redundant_placeholder2P
Lsimple_rnn_50_while_simple_rnn_50_while_cond_190489___redundant_placeholder3 
simple_rnn_50_while_identity

simple_rnn_50/while/LessLesssimple_rnn_50_while_placeholder6simple_rnn_50_while_less_simple_rnn_50_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_50/while/IdentityIdentitysimple_rnn_50/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_50_while_identity%simple_rnn_50/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¾@
Æ
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191927

inputsD
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191861*
condR
while_cond_191860*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
+__inference_dropout_28_layer_call_fn_191418

inputs
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_190098s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
7
¤
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189606

inputs,
simple_rnn_cell_51_189531:	@à(
simple_rnn_cell_51_189533:	à-
simple_rnn_cell_51_189535:
àà
identity¢*simple_rnn_cell_51/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskñ
*simple_rnn_cell_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_51_189531simple_rnn_cell_51_189533simple_rnn_cell_51_189535*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189487n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_51_189531simple_rnn_cell_51_189533simple_rnn_cell_51_189535*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189543*
condR
while_cond_189542*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà{
NoOpNoOp+^simple_rnn_cell_51/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 2X
*simple_rnn_cell_51/StatefulPartitionedCall*simple_rnn_cell_51/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾@
Æ
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_190069

inputsD
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_190003*
condR
while_cond_190002*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á@
Ä
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191184
inputs_0C
1simple_rnn_cell_50_matmul_readvariableop_resource:@@
2simple_rnn_cell_50_biasadd_readvariableop_resource:@E
3simple_rnn_cell_50_matmul_1_readvariableop_resource:@@
identity¢)simple_rnn_cell_50/BiasAdd/ReadVariableOp¢(simple_rnn_cell_50/MatMul/ReadVariableOp¢*simple_rnn_cell_50/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¡
simple_rnn_cell_50/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¯
simple_rnn_cell_50/BiasAddBiasAdd#simple_rnn_cell_50/MatMul:product:01simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_50/MatMul_1MatMulzeros:output:02simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
simple_rnn_cell_50/addAddV2#simple_rnn_cell_50/BiasAdd:output:0%simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
simple_rnn_cell_50/TanhTanhsimple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_50_matmul_readvariableop_resource2simple_rnn_cell_50_biasadd_readvariableop_resource3simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191118*
condR
while_cond_191117*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ò
NoOpNoOp*^simple_rnn_cell_50/BiasAdd/ReadVariableOp)^simple_rnn_cell_50/MatMul/ReadVariableOp+^simple_rnn_cell_50/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2V
)simple_rnn_cell_50/BiasAdd/ReadVariableOp)simple_rnn_cell_50/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_50/MatMul/ReadVariableOp(simple_rnn_cell_50/MatMul/ReadVariableOp2X
*simple_rnn_cell_50/MatMul_1/ReadVariableOp*simple_rnn_cell_50/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
7
¤
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_189443

inputs,
simple_rnn_cell_51_189368:	@à(
simple_rnn_cell_51_189370:	à-
simple_rnn_cell_51_189372:
àà
identity¢*simple_rnn_cell_51/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskñ
*simple_rnn_cell_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_51_189368simple_rnn_cell_51_189370simple_rnn_cell_51_189372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_189367n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_51_189368simple_rnn_cell_51_189370simple_rnn_cell_51_189372*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_189380*
condR
while_cond_189379*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà{
NoOpNoOp+^simple_rnn_cell_51/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 2X
*simple_rnn_cell_51/StatefulPartitionedCall*simple_rnn_cell_51/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼Ã
ü

!__inference__wrapped_model_189015
simple_rnn_50_input_
Msequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource:@\
Nsequential_25_simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource:@a
Osequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@`
Msequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource:	@à]
Nsequential_25_simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource:	àc
Osequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource:
ààH
5sequential_25_dense_25_matmul_readvariableop_resource:	àD
6sequential_25_dense_25_biasadd_readvariableop_resource:
identity¢-sequential_25/dense_25/BiasAdd/ReadVariableOp¢,sequential_25/dense_25/MatMul/ReadVariableOp¢Esequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢Dsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp¢Fsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp¢!sequential_25/simple_rnn_50/while¢Esequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢Dsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp¢Fsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp¢!sequential_25/simple_rnn_51/whiled
!sequential_25/simple_rnn_50/ShapeShapesimple_rnn_50_input*
T0*
_output_shapes
:y
/sequential_25/simple_rnn_50/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_25/simple_rnn_50/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_25/simple_rnn_50/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_25/simple_rnn_50/strided_sliceStridedSlice*sequential_25/simple_rnn_50/Shape:output:08sequential_25/simple_rnn_50/strided_slice/stack:output:0:sequential_25/simple_rnn_50/strided_slice/stack_1:output:0:sequential_25/simple_rnn_50/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_25/simple_rnn_50/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@³
%sequential_25/simple_rnn_50/zeros/mulMul2sequential_25/simple_rnn_50/strided_slice:output:00sequential_25/simple_rnn_50/zeros/mul/y:output:0*
T0*
_output_shapes
: k
(sequential_25/simple_rnn_50/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è­
&sequential_25/simple_rnn_50/zeros/LessLess)sequential_25/simple_rnn_50/zeros/mul:z:01sequential_25/simple_rnn_50/zeros/Less/y:output:0*
T0*
_output_shapes
: l
*sequential_25/simple_rnn_50/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ç
(sequential_25/simple_rnn_50/zeros/packedPack2sequential_25/simple_rnn_50/strided_slice:output:03sequential_25/simple_rnn_50/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_25/simple_rnn_50/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    À
!sequential_25/simple_rnn_50/zerosFill1sequential_25/simple_rnn_50/zeros/packed:output:00sequential_25/simple_rnn_50/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_25/simple_rnn_50/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ²
%sequential_25/simple_rnn_50/transpose	Transposesimple_rnn_50_input3sequential_25/simple_rnn_50/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
#sequential_25/simple_rnn_50/Shape_1Shape)sequential_25/simple_rnn_50/transpose:y:0*
T0*
_output_shapes
:{
1sequential_25/simple_rnn_50/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_50/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_25/simple_rnn_50/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+sequential_25/simple_rnn_50/strided_slice_1StridedSlice,sequential_25/simple_rnn_50/Shape_1:output:0:sequential_25/simple_rnn_50/strided_slice_1/stack:output:0<sequential_25/simple_rnn_50/strided_slice_1/stack_1:output:0<sequential_25/simple_rnn_50/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7sequential_25/simple_rnn_50/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_25/simple_rnn_50/TensorArrayV2TensorListReserve@sequential_25/simple_rnn_50/TensorArrayV2/element_shape:output:04sequential_25/simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¢
Qsequential_25/simple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Csequential_25/simple_rnn_50/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_25/simple_rnn_50/transpose:y:0Zsequential_25/simple_rnn_50/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
1sequential_25/simple_rnn_50/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_50/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_25/simple_rnn_50/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
+sequential_25/simple_rnn_50/strided_slice_2StridedSlice)sequential_25/simple_rnn_50/transpose:y:0:sequential_25/simple_rnn_50/strided_slice_2/stack:output:0<sequential_25/simple_rnn_50/strided_slice_2/stack_1:output:0<sequential_25/simple_rnn_50/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÒ
Dsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOpMsequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0õ
5sequential_25/simple_rnn_50/simple_rnn_cell_50/MatMulMatMul4sequential_25/simple_rnn_50/strided_slice_2:output:0Lsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
Esequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOpNsequential_25_simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
6sequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAddBiasAdd?sequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul:product:0Msequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
Fsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOpOsequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0ï
7sequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1MatMul*sequential_25/simple_rnn_50/zeros:output:0Nsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
2sequential_25/simple_rnn_50/simple_rnn_cell_50/addAddV2?sequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd:output:0Asequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
3sequential_25/simple_rnn_50/simple_rnn_cell_50/TanhTanh6sequential_25/simple_rnn_50/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
9sequential_25/simple_rnn_50/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
+sequential_25/simple_rnn_50/TensorArrayV2_1TensorListReserveBsequential_25/simple_rnn_50/TensorArrayV2_1/element_shape:output:04sequential_25/simple_rnn_50/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒb
 sequential_25/simple_rnn_50/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_25/simple_rnn_50/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿp
.sequential_25/simple_rnn_50/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ç
!sequential_25/simple_rnn_50/whileWhile7sequential_25/simple_rnn_50/while/loop_counter:output:0=sequential_25/simple_rnn_50/while/maximum_iterations:output:0)sequential_25/simple_rnn_50/time:output:04sequential_25/simple_rnn_50/TensorArrayV2_1:handle:0*sequential_25/simple_rnn_50/zeros:output:04sequential_25/simple_rnn_50/strided_slice_1:output:0Ssequential_25/simple_rnn_50/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_readvariableop_resourceNsequential_25_simple_rnn_50_simple_rnn_cell_50_biasadd_readvariableop_resourceOsequential_25_simple_rnn_50_simple_rnn_cell_50_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_25_simple_rnn_50_while_body_188832*9
cond1R/
-sequential_25_simple_rnn_50_while_cond_188831*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Lsequential_25/simple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
>sequential_25/simple_rnn_50/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_25/simple_rnn_50/while:output:3Usequential_25/simple_rnn_50/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0
1sequential_25/simple_rnn_50/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ}
3sequential_25/simple_rnn_50/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_50/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+sequential_25/simple_rnn_50/strided_slice_3StridedSliceGsequential_25/simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_25/simple_rnn_50/strided_slice_3/stack:output:0<sequential_25/simple_rnn_50/strided_slice_3/stack_1:output:0<sequential_25/simple_rnn_50/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
,sequential_25/simple_rnn_50/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ê
'sequential_25/simple_rnn_50/transpose_1	TransposeGsequential_25/simple_rnn_50/TensorArrayV2Stack/TensorListStack:tensor:05sequential_25/simple_rnn_50/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!sequential_25/dropout_28/IdentityIdentity+sequential_25/simple_rnn_50/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
!sequential_25/simple_rnn_51/ShapeShape*sequential_25/dropout_28/Identity:output:0*
T0*
_output_shapes
:y
/sequential_25/simple_rnn_51/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_25/simple_rnn_51/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_25/simple_rnn_51/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_25/simple_rnn_51/strided_sliceStridedSlice*sequential_25/simple_rnn_51/Shape:output:08sequential_25/simple_rnn_51/strided_slice/stack:output:0:sequential_25/simple_rnn_51/strided_slice/stack_1:output:0:sequential_25/simple_rnn_51/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'sequential_25/simple_rnn_51/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à³
%sequential_25/simple_rnn_51/zeros/mulMul2sequential_25/simple_rnn_51/strided_slice:output:00sequential_25/simple_rnn_51/zeros/mul/y:output:0*
T0*
_output_shapes
: k
(sequential_25/simple_rnn_51/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è­
&sequential_25/simple_rnn_51/zeros/LessLess)sequential_25/simple_rnn_51/zeros/mul:z:01sequential_25/simple_rnn_51/zeros/Less/y:output:0*
T0*
_output_shapes
: m
*sequential_25/simple_rnn_51/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :àÇ
(sequential_25/simple_rnn_51/zeros/packedPack2sequential_25/simple_rnn_51/strided_slice:output:03sequential_25/simple_rnn_51/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_25/simple_rnn_51/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Á
!sequential_25/simple_rnn_51/zerosFill1sequential_25/simple_rnn_51/zeros/packed:output:00sequential_25/simple_rnn_51/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
*sequential_25/simple_rnn_51/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
%sequential_25/simple_rnn_51/transpose	Transpose*sequential_25/dropout_28/Identity:output:03sequential_25/simple_rnn_51/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
#sequential_25/simple_rnn_51/Shape_1Shape)sequential_25/simple_rnn_51/transpose:y:0*
T0*
_output_shapes
:{
1sequential_25/simple_rnn_51/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_51/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_25/simple_rnn_51/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+sequential_25/simple_rnn_51/strided_slice_1StridedSlice,sequential_25/simple_rnn_51/Shape_1:output:0:sequential_25/simple_rnn_51/strided_slice_1/stack:output:0<sequential_25/simple_rnn_51/strided_slice_1/stack_1:output:0<sequential_25/simple_rnn_51/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7sequential_25/simple_rnn_51/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_25/simple_rnn_51/TensorArrayV2TensorListReserve@sequential_25/simple_rnn_51/TensorArrayV2/element_shape:output:04sequential_25/simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¢
Qsequential_25/simple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ´
Csequential_25/simple_rnn_51/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_25/simple_rnn_51/transpose:y:0Zsequential_25/simple_rnn_51/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
1sequential_25/simple_rnn_51/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_51/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_25/simple_rnn_51/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
+sequential_25/simple_rnn_51/strided_slice_2StridedSlice)sequential_25/simple_rnn_51/transpose:y:0:sequential_25/simple_rnn_51/strided_slice_2/stack:output:0<sequential_25/simple_rnn_51/strided_slice_2/stack_1:output:0<sequential_25/simple_rnn_51/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskÓ
Dsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOpMsequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0ö
5sequential_25/simple_rnn_51/simple_rnn_cell_51/MatMulMatMul4sequential_25/simple_rnn_51/strided_slice_2:output:0Lsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÑ
Esequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOpNsequential_25_simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0
6sequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAddBiasAdd?sequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul:product:0Msequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàØ
Fsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpOsequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0ð
7sequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1MatMul*sequential_25/simple_rnn_51/zeros:output:0Nsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàò
2sequential_25/simple_rnn_51/simple_rnn_cell_51/addAddV2?sequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd:output:0Asequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà¦
3sequential_25/simple_rnn_51/simple_rnn_cell_51/TanhTanh6sequential_25/simple_rnn_51/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
9sequential_25/simple_rnn_51/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   
+sequential_25/simple_rnn_51/TensorArrayV2_1TensorListReserveBsequential_25/simple_rnn_51/TensorArrayV2_1/element_shape:output:04sequential_25/simple_rnn_51/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒb
 sequential_25/simple_rnn_51/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_25/simple_rnn_51/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿp
.sequential_25/simple_rnn_51/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : É
!sequential_25/simple_rnn_51/whileWhile7sequential_25/simple_rnn_51/while/loop_counter:output:0=sequential_25/simple_rnn_51/while/maximum_iterations:output:0)sequential_25/simple_rnn_51/time:output:04sequential_25/simple_rnn_51/TensorArrayV2_1:handle:0*sequential_25/simple_rnn_51/zeros:output:04sequential_25/simple_rnn_51/strided_slice_1:output:0Ssequential_25/simple_rnn_51/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_readvariableop_resourceNsequential_25_simple_rnn_51_simple_rnn_cell_51_biasadd_readvariableop_resourceOsequential_25_simple_rnn_51_simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_25_simple_rnn_51_while_body_188941*9
cond1R/
-sequential_25_simple_rnn_51_while_cond_188940*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
Lsequential_25/simple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   
>sequential_25/simple_rnn_51/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_25/simple_rnn_51/while:output:3Usequential_25/simple_rnn_51/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
element_dtype0
1sequential_25/simple_rnn_51/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ}
3sequential_25/simple_rnn_51/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_25/simple_rnn_51/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+sequential_25/simple_rnn_51/strided_slice_3StridedSliceGsequential_25/simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_25/simple_rnn_51/strided_slice_3/stack:output:0<sequential_25/simple_rnn_51/strided_slice_3/stack_1:output:0<sequential_25/simple_rnn_51/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_mask
,sequential_25/simple_rnn_51/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ë
'sequential_25/simple_rnn_51/transpose_1	TransposeGsequential_25/simple_rnn_51/TensorArrayV2Stack/TensorListStack:tensor:05sequential_25/simple_rnn_51/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
!sequential_25/dropout_29/IdentityIdentity4sequential_25/simple_rnn_51/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà£
,sequential_25/dense_25/MatMul/ReadVariableOpReadVariableOp5sequential_25_dense_25_matmul_readvariableop_resource*
_output_shapes
:	à*
dtype0»
sequential_25/dense_25/MatMulMatMul*sequential_25/dropout_29/Identity:output:04sequential_25/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_25/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_25_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_25/dense_25/BiasAddBiasAdd'sequential_25/dense_25/MatMul:product:05sequential_25/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_25/dense_25/SoftmaxSoftmax'sequential_25/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_25/dense_25/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp.^sequential_25/dense_25/BiasAdd/ReadVariableOp-^sequential_25/dense_25/MatMul/ReadVariableOpF^sequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOpE^sequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOpG^sequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp"^sequential_25/simple_rnn_50/whileF^sequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOpE^sequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOpG^sequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp"^sequential_25/simple_rnn_51/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2^
-sequential_25/dense_25/BiasAdd/ReadVariableOp-sequential_25/dense_25/BiasAdd/ReadVariableOp2\
,sequential_25/dense_25/MatMul/ReadVariableOp,sequential_25/dense_25/MatMul/ReadVariableOp2
Esequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOpEsequential_25/simple_rnn_50/simple_rnn_cell_50/BiasAdd/ReadVariableOp2
Dsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOpDsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul/ReadVariableOp2
Fsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOpFsequential_25/simple_rnn_50/simple_rnn_cell_50/MatMul_1/ReadVariableOp2F
!sequential_25/simple_rnn_50/while!sequential_25/simple_rnn_50/while2
Esequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOpEsequential_25/simple_rnn_51/simple_rnn_cell_51/BiasAdd/ReadVariableOp2
Dsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOpDsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul/ReadVariableOp2
Fsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOpFsequential_25/simple_rnn_51/simple_rnn_cell_51/MatMul_1/ReadVariableOp2F
!sequential_25/simple_rnn_51/while!sequential_25/simple_rnn_51/while:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input
Ü
ª
while_cond_191524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_191524___redundant_placeholder04
0while_while_cond_191524___redundant_placeholder14
0while_while_cond_191524___redundant_placeholder24
0while_while_cond_191524___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿà: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
:
ó@
È
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191591
inputs_0D
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191525*
condR
while_cond_191524*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0

¸
.__inference_simple_rnn_50_layer_call_fn_190949

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_189733s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó@
È
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191703
inputs_0D
1simple_rnn_cell_51_matmul_readvariableop_resource:	@àA
2simple_rnn_cell_51_biasadd_readvariableop_resource:	àG
3simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà
identity¢)simple_rnn_cell_51/BiasAdd/ReadVariableOp¢(simple_rnn_cell_51/MatMul/ReadVariableOp¢*simple_rnn_cell_51/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :à_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :às
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
(simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_51_matmul_readvariableop_resource*
_output_shapes
:	@à*
dtype0¢
simple_rnn_cell_51/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
)simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_51_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype0°
simple_rnn_cell_51/BiasAddBiasAdd#simple_rnn_cell_51/MatMul:product:01simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà 
*simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_51_matmul_1_readvariableop_resource* 
_output_shapes
:
àà*
dtype0
simple_rnn_cell_51/MatMul_1MatMulzeros:output:02simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_cell_51/addAddV2#simple_rnn_cell_51/BiasAdd:output:0%simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
simple_rnn_cell_51/TanhTanhsimple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_51_matmul_readvariableop_resource2simple_rnn_cell_51_biasadd_readvariableop_resource3simple_rnn_cell_51_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_191637*
condR
while_cond_191636*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿàh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÒ
NoOpNoOp*^simple_rnn_cell_51/BiasAdd/ReadVariableOp)^simple_rnn_cell_51/MatMul/ReadVariableOp+^simple_rnn_cell_51/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : 2V
)simple_rnn_cell_51/BiasAdd/ReadVariableOp)simple_rnn_cell_51/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_51/MatMul/ReadVariableOp(simple_rnn_cell_51/MatMul/ReadVariableOp2X
*simple_rnn_cell_51/MatMul_1/ReadVariableOp*simple_rnn_cell_51/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0
¾

Û
3__inference_simple_rnn_cell_50_layer_call_fn_191988

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_189067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
ê,
Ù
while_body_191525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àI
:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	àO
;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àG
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:	àM
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_51/MatMul/ReadVariableOp¢0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0©
.while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0Æ
while/simple_rnn_cell_51/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà§
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0Â
 while/simple_rnn_cell_51/BiasAddBiasAdd)while/simple_rnn_cell_51/MatMul:product:07while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà®
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0­
!while/simple_rnn_cell_51/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà°
while/simple_rnn_cell_51/addAddV2)while/simple_rnn_cell_51/BiasAdd:output:0+while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàz
while/simple_rnn_cell_51/TanhTanh while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity!while/simple_rnn_cell_51/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàâ

while/NoOpNoOp0^while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_51/MatMul/ReadVariableOp1^while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_51_biasadd_readvariableop_resource:while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_51_matmul_1_readvariableop_resource;while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_51_matmul_readvariableop_resource9while_simple_rnn_cell_51_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2b
/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_51/MatMul/ReadVariableOp.while/simple_rnn_cell_51/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp0while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 
÷9
ý
simple_rnn_51_while_body_1908358
4simple_rnn_51_while_simple_rnn_51_while_loop_counter>
:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations#
simple_rnn_51_while_placeholder%
!simple_rnn_51_while_placeholder_1%
!simple_rnn_51_while_placeholder_27
3simple_rnn_51_while_simple_rnn_51_strided_slice_1_0s
osimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0Z
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0:	@àW
Hsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0:	à]
Isimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0:
àà 
simple_rnn_51_while_identity"
simple_rnn_51_while_identity_1"
simple_rnn_51_while_identity_2"
simple_rnn_51_while_identity_3"
simple_rnn_51_while_identity_45
1simple_rnn_51_while_simple_rnn_51_strided_slice_1q
msimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorX
Esimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource:	@àU
Fsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource:	à[
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource:
àà¢=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp¢<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp¢>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp
Esimple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ì
7simple_rnn_51/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_51_while_placeholderNsimple_rnn_51/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
element_dtype0Å
<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0*
_output_shapes
:	@à*
dtype0ð
-simple_rnn_51/while/simple_rnn_cell_51/MatMulMatMul>simple_rnn_51/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÃ
=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype0ì
.simple_rnn_51/while/simple_rnn_cell_51/BiasAddBiasAdd7simple_rnn_51/while/simple_rnn_cell_51/MatMul:product:0Esimple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÊ
>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0* 
_output_shapes
:
àà*
dtype0×
/simple_rnn_51/while/simple_rnn_cell_51/MatMul_1MatMul!simple_rnn_51_while_placeholder_2Fsimple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàÚ
*simple_rnn_51/while/simple_rnn_cell_51/addAddV27simple_rnn_51/while/simple_rnn_cell_51/BiasAdd:output:09simple_rnn_51/while/simple_rnn_cell_51/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
+simple_rnn_51/while/simple_rnn_cell_51/TanhTanh.simple_rnn_51/while/simple_rnn_cell_51/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
8simple_rnn_51/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_51_while_placeholder_1simple_rnn_51_while_placeholder/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒ[
simple_rnn_51/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_51/while/addAddV2simple_rnn_51_while_placeholder"simple_rnn_51/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_51/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_51/while/add_1AddV24simple_rnn_51_while_simple_rnn_51_while_loop_counter$simple_rnn_51/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_51/while/IdentityIdentitysimple_rnn_51/while/add_1:z:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: ¢
simple_rnn_51/while/Identity_1Identity:simple_rnn_51_while_simple_rnn_51_while_maximum_iterations^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_51/while/Identity_2Identitysimple_rnn_51/while/add:z:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: °
simple_rnn_51/while/Identity_3IdentityHsimple_rnn_51/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_51/while/NoOp*
T0*
_output_shapes
: ©
simple_rnn_51/while/Identity_4Identity/simple_rnn_51/while/simple_rnn_cell_51/Tanh:y:0^simple_rnn_51/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
simple_rnn_51/while/NoOpNoOp>^simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp=^simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp?^simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_51_while_identity%simple_rnn_51/while/Identity:output:0"I
simple_rnn_51_while_identity_1'simple_rnn_51/while/Identity_1:output:0"I
simple_rnn_51_while_identity_2'simple_rnn_51/while/Identity_2:output:0"I
simple_rnn_51_while_identity_3'simple_rnn_51/while/Identity_3:output:0"I
simple_rnn_51_while_identity_4'simple_rnn_51/while/Identity_4:output:0"h
1simple_rnn_51_while_simple_rnn_51_strided_slice_13simple_rnn_51_while_simple_rnn_51_strided_slice_1_0"
Fsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resourceHsimple_rnn_51_while_simple_rnn_cell_51_biasadd_readvariableop_resource_0"
Gsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resourceIsimple_rnn_51_while_simple_rnn_cell_51_matmul_1_readvariableop_resource_0"
Esimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resourceGsimple_rnn_51_while_simple_rnn_cell_51_matmul_readvariableop_resource_0"à
msimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensorosimple_rnn_51_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_51_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿà: : : : : 2~
=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp=simple_rnn_51/while/simple_rnn_cell_51/BiasAdd/ReadVariableOp2|
<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp<simple_rnn_51/while/simple_rnn_cell_51/MatMul/ReadVariableOp2
>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp>simple_rnn_51/while/simple_rnn_cell_51/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà:

_output_shapes
: :

_output_shapes
: 


e
F__inference_dropout_28_layer_call_and_return_conditional_losses_191435

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È

)__inference_dense_25_layer_call_fn_191963

inputs
unknown:	à
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_189885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs

ë
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192036

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0
Ö,
Ñ
while_body_190160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_50_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_50_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource:@@¢/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp¢.while/simple_rnn_cell_50/MatMul/ReadVariableOp¢0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.while/simple_rnn_cell_50/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_50_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Å
while/simple_rnn_cell_50/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Á
 while/simple_rnn_cell_50/BiasAddBiasAdd)while/simple_rnn_cell_50/MatMul:product:07while/simple_rnn_cell_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0¬
!while/simple_rnn_cell_50/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_50/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
while/simple_rnn_cell_50/addAddV2)while/simple_rnn_cell_50/BiasAdd:output:0+while/simple_rnn_cell_50/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
while/simple_rnn_cell_50/TanhTanh while/simple_rnn_cell_50/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_50/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_50/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â

while/NoOpNoOp0^while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_50/MatMul/ReadVariableOp1^while/simple_rnn_cell_50/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_50_biasadd_readvariableop_resource:while_simple_rnn_cell_50_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_50_matmul_1_readvariableop_resource;while_simple_rnn_cell_50_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_50_matmul_readvariableop_resource9while_simple_rnn_cell_50_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ@: : : : : 2b
/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp/while/simple_rnn_cell_50/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_50/MatMul/ReadVariableOp.while/simple_rnn_cell_50/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp0while/simple_rnn_cell_50/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ú
d
+__inference_dropout_29_layer_call_fn_191937

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189941p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿà22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ù
á
I__inference_sequential_25_layer_call_and_return_conditional_losses_190373
simple_rnn_50_input&
simple_rnn_50_190351:@"
simple_rnn_50_190353:@&
simple_rnn_50_190355:@@'
simple_rnn_51_190359:	@à#
simple_rnn_51_190361:	à(
simple_rnn_51_190363:
àà"
dense_25_190367:	à
dense_25_190369:
identity¢ dense_25/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢%simple_rnn_50/StatefulPartitionedCall¢%simple_rnn_51/StatefulPartitionedCall°
%simple_rnn_50/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_50_inputsimple_rnn_50_190351simple_rnn_50_190353simple_rnn_50_190355*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_190226û
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_28_layer_call_and_return_conditional_losses_190098Å
%simple_rnn_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0simple_rnn_51_190359simple_rnn_51_190361simple_rnn_51_190363*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_190069
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_51/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_29_layer_call_and_return_conditional_losses_189941
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0dense_25_190367dense_25_190369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_189885x
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_25/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall&^simple_rnn_50/StatefulPartitionedCall&^simple_rnn_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2N
%simple_rnn_50/StatefulPartitionedCall%simple_rnn_50/StatefulPartitionedCall2N
%simple_rnn_51/StatefulPartitionedCall%simple_rnn_51/StatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesimple_rnn_50_input"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
W
simple_rnn_50_input@
%serving_default_simple_rnn_50_input:0ÿÿÿÿÿÿÿÿÿ<
dense_250
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:­¸
ö
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
~__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
Å
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ã
&iter

'beta_1

(beta_2
	)decay
*learning_rate mn!mo+mp,mq-mr.ms/mt0mu vv!vw+vx,vy-vz.v{/v|0v}"
	optimizer
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ë
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
	regularization_losses
~__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
Ó

+kernel
,recurrent_kernel
-bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
¼

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó

.kernel
/recurrent_kernel
0bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
¼

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	à2dense_25/kernel
:2dense_25/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
"	variables
#trainable_variables
$regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
9:7@2'simple_rnn_50/simple_rnn_cell_50/kernel
C:A@@21simple_rnn_50/simple_rnn_cell_50/recurrent_kernel
3:1@2%simple_rnn_50/simple_rnn_cell_50/bias
::8	@à2'simple_rnn_51/simple_rnn_cell_51/kernel
E:C
àà21simple_rnn_51/simple_rnn_cell_51/recurrent_kernel
4:2à2%simple_rnn_51/simple_rnn_cell_51/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
6	variables
7trainable_variables
8regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
°
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	etotal
	fcount
g	variables
h	keras_api"
_tf_keras_metric
^
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
':%	à2Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
>:<@2.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/m
H:F@@28Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/m
8:6@2,Adam/simple_rnn_50/simple_rnn_cell_50/bias/m
?:=	@à2.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/m
J:H
àà28Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/m
9:7à2,Adam/simple_rnn_51/simple_rnn_cell_51/bias/m
':%	à2Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
>:<@2.Adam/simple_rnn_50/simple_rnn_cell_50/kernel/v
H:F@@28Adam/simple_rnn_50/simple_rnn_cell_50/recurrent_kernel/v
8:6@2,Adam/simple_rnn_50/simple_rnn_cell_50/bias/v
?:=	@à2.Adam/simple_rnn_51/simple_rnn_cell_51/kernel/v
J:H
àà28Adam/simple_rnn_51/simple_rnn_cell_51/recurrent_kernel/v
9:7à2,Adam/simple_rnn_51/simple_rnn_cell_51/bias/v
2
.__inference_sequential_25_layer_call_fn_189911
.__inference_sequential_25_layer_call_fn_190423
.__inference_sequential_25_layer_call_fn_190444
.__inference_sequential_25_layer_call_fn_190323À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_25_layer_call_and_return_conditional_losses_190673
I__inference_sequential_25_layer_call_and_return_conditional_losses_190916
I__inference_sequential_25_layer_call_and_return_conditional_losses_190348
I__inference_sequential_25_layer_call_and_return_conditional_losses_190373À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ØBÕ
!__inference__wrapped_model_189015simple_rnn_50_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_simple_rnn_50_layer_call_fn_190927
.__inference_simple_rnn_50_layer_call_fn_190938
.__inference_simple_rnn_50_layer_call_fn_190949
.__inference_simple_rnn_50_layer_call_fn_190960Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191072
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191184
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191296
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191408Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_28_layer_call_fn_191413
+__inference_dropout_28_layer_call_fn_191418´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_28_layer_call_and_return_conditional_losses_191423
F__inference_dropout_28_layer_call_and_return_conditional_losses_191435´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_simple_rnn_51_layer_call_fn_191446
.__inference_simple_rnn_51_layer_call_fn_191457
.__inference_simple_rnn_51_layer_call_fn_191468
.__inference_simple_rnn_51_layer_call_fn_191479Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191591
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191703
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191815
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191927Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_29_layer_call_fn_191932
+__inference_dropout_29_layer_call_fn_191937´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_29_layer_call_and_return_conditional_losses_191942
F__inference_dropout_29_layer_call_and_return_conditional_losses_191954´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_25_layer_call_fn_191963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_25_layer_call_and_return_conditional_losses_191974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×BÔ
$__inference_signature_wrapper_190402simple_rnn_50_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
3__inference_simple_rnn_cell_50_layer_call_fn_191988
3__inference_simple_rnn_cell_50_layer_call_fn_192002¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192019
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192036¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
®2«
3__inference_simple_rnn_cell_51_layer_call_fn_192050
3__inference_simple_rnn_cell_51_layer_call_fn_192064¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192081
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192098¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 §
!__inference__wrapped_model_189015+-,.0/ !@¢=
6¢3
1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_25"
dense_25ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_25_layer_call_and_return_conditional_losses_191974] !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿà
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_25_layer_call_fn_191963P !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿà
ª "ÿÿÿÿÿÿÿÿÿ®
F__inference_dropout_28_layer_call_and_return_conditional_losses_191423d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 ®
F__inference_dropout_28_layer_call_and_return_conditional_losses_191435d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ@
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_dropout_28_layer_call_fn_191413W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@
+__inference_dropout_28_layer_call_fn_191418W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¨
F__inference_dropout_29_layer_call_and_return_conditional_losses_191942^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿà
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 ¨
F__inference_dropout_29_layer_call_and_return_conditional_losses_191954^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿà
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 
+__inference_dropout_29_layer_call_fn_191932Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿà
p 
ª "ÿÿÿÿÿÿÿÿÿà
+__inference_dropout_29_layer_call_fn_191937Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿà
p
ª "ÿÿÿÿÿÿÿÿÿàÈ
I__inference_sequential_25_layer_call_and_return_conditional_losses_190348{+-,.0/ !H¢E
>¢;
1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
I__inference_sequential_25_layer_call_and_return_conditional_losses_190373{+-,.0/ !H¢E
>¢;
1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
I__inference_sequential_25_layer_call_and_return_conditional_losses_190673n+-,.0/ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
I__inference_sequential_25_layer_call_and_return_conditional_losses_190916n+-,.0/ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
.__inference_sequential_25_layer_call_fn_189911n+-,.0/ !H¢E
>¢;
1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
.__inference_sequential_25_layer_call_fn_190323n+-,.0/ !H¢E
>¢;
1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_25_layer_call_fn_190423a+-,.0/ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_25_layer_call_fn_190444a+-,.0/ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÁ
$__inference_signature_wrapper_190402+-,.0/ !W¢T
¢ 
MªJ
H
simple_rnn_50_input1.
simple_rnn_50_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_25"
dense_25ÿÿÿÿÿÿÿÿÿØ
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191072+-,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ø
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191184+-,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¾
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191296q+-,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 ¾
I__inference_simple_rnn_50_layer_call_and_return_conditional_losses_191408q+-,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 ¯
.__inference_simple_rnn_50_layer_call_fn_190927}+-,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¯
.__inference_simple_rnn_50_layer_call_fn_190938}+-,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
.__inference_simple_rnn_50_layer_call_fn_190949d+-,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
.__inference_simple_rnn_50_layer_call_fn_190960d+-,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@Ë
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191591~.0/O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 Ë
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191703~.0/O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 »
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191815n.0/?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 »
I__inference_simple_rnn_51_layer_call_and_return_conditional_losses_191927n.0/?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 £
.__inference_simple_rnn_51_layer_call_fn_191446q.0/O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿà£
.__inference_simple_rnn_51_layer_call_fn_191457q.0/O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

 
p

 
ª "ÿÿÿÿÿÿÿÿÿà
.__inference_simple_rnn_51_layer_call_fn_191468a.0/?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿà
.__inference_simple_rnn_51_layer_call_fn_191479a.0/?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ@

 
p

 
ª "ÿÿÿÿÿÿÿÿÿà
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192019·+-,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 
N__inference_simple_rnn_cell_50_layer_call_and_return_conditional_losses_192036·+-,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ@
$!

0/1/0ÿÿÿÿÿÿÿÿÿ@
 á
3__inference_simple_rnn_cell_50_layer_call_fn_191988©+-,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@á
3__inference_simple_rnn_cell_50_layer_call_fn_192002©+-,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ@
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ@
"

1/0ÿÿÿÿÿÿÿÿÿ@
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192081º.0/]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ@
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿà
p 
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿà
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿà
 
N__inference_simple_rnn_cell_51_layer_call_and_return_conditional_losses_192098º.0/]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ@
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿà
p
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿà
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿà
 ä
3__inference_simple_rnn_cell_51_layer_call_fn_192050¬.0/]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ@
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿà
p 
ª "F¢C

0ÿÿÿÿÿÿÿÿÿà
# 

1/0ÿÿÿÿÿÿÿÿÿàä
3__inference_simple_rnn_cell_51_layer_call_fn_192064¬.0/]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ@
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿà
p
ª "F¢C

0ÿÿÿÿÿÿÿÿÿà
# 

1/0ÿÿÿÿÿÿÿÿÿà