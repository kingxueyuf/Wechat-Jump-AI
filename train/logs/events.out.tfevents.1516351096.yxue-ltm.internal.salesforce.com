       ŁK"	   kÖAbrain.Event:2ńŘ      ŽLY8	+Á kÖA"óý
z
imgPlaceholder*
dtype0*&
shape:˙˙˙˙˙˙˙˙˙*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
labelPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
is_trainingPlaceholder*
dtype0
*
shape:*
_output_shapes
:
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
G
lrPlaceholder*
dtype0*
shape:*
_output_shapes
:
š
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
¤
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
Ś
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  ?*
_output_shapes
: 

Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w

5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:

1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Á
coarse/conv1/conv1-w
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ń
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:

&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
Š
coarse/conv1/conv1-b
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
Ú
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
á
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ç
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC

coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
¸
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ť
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
ť
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Š
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Ő
coarse/conv2-conv/conv2-conv-w
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 

%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
ł
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
°
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
˝
coarse/conv2-conv/conv2-conv-b
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
§
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 

3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ď
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
data_formatNHWC

&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  ?*
_output_shapes
: 
Ť
coarse/conv2-bn/gamma
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
Ý
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 

&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
Š
coarse/conv2-bn/beta
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
Ú
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Ş
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
ˇ
coarse/conv2-bn/moving_mean
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *.
_class$
" loc:@coarse/conv2-bn/moving_mean*
shared_name 
ö
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 

 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ą
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  ?*
_output_shapes
: 
ż
coarse/conv2-bn/moving_variance
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
shared_name 

&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
Ş
$coarse/conv2-bn/moving_variance/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
m
"coarse/coarse/conv2-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv2-bn/cond/switch_tIdentity$coarse/coarse/conv2-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv2-bn/cond/switch_fIdentity"coarse/coarse/conv2-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv2-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ž
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ó
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
Ń
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
ˇ
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%o:*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
T0*
is_training(*
data_formatNHWC
Ŕ
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ő
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
Ó
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
á
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
é
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
ß
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
T0*
is_training( *
data_formatNHWC
Ë
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : *
T0*
N
ş
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
_output_shapes

: : *
T0*
N
ş
#coarse/coarse/conv2-bn/cond/Merge_2Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2*
_output_shapes

: : *
T0*
N
l
'coarse/coarse/conv2-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv2-bn/ExpandDims
ExpandDims'coarse/coarse/conv2-bn/ExpandDims/input%coarse/coarse/conv2-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv2-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv2-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv2-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv2-bn/ExpandDims_1/input'coarse/coarse/conv2-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv2-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
´
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Š
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
Ř
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
Ň
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ä
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
ł
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
ŕ
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
Ú
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
đ
(coarse/coarse/conv2-bn/AssignMovingAvg_1	AssignSubcoarse/conv2-bn/moving_variance,coarse/coarse/conv2-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking( *
T0*
_output_shapes
: 

coarse/coarse/conv2-reluRelu!coarse/coarse/conv2-bn/cond/Merge*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ć
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Í
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
¸
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ť
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
ť
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Š
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Ő
coarse/conv3-conv/conv3-conv-w
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 

%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
ł
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
°
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
˝
coarse/conv3-conv/conv3-conv-b
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
§
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@

3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
í
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
T0*
data_formatNHWC

&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  ?*
_output_shapes
:@
Ť
coarse/conv3-bn/gamma
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
Ý
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@

&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
Š
coarse/conv3-bn/beta
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
Ú
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ş
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
ˇ
coarse/conv3-bn/moving_mean
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
shared_name 
ö
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@

 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ą
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  ?*
_output_shapes
:@
ż
coarse/conv3-bn/moving_variance
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
shared_name 

&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
Ş
$coarse/conv3-bn/moving_variance/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
m
"coarse/coarse/conv3-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv3-bn/cond/switch_tIdentity$coarse/coarse/conv3-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv3-bn/cond/switch_fIdentity"coarse/coarse/conv3-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv3-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ş
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
Ó
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
Ń
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
ľ
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@*
T0*
is_training(*
data_formatNHWC
ź
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
Ő
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
Ó
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
á
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
é
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
Ý
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
É
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: *
T0*
N
ş
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
_output_shapes

:@: *
T0*
N
ş
#coarse/coarse/conv3-bn/cond/Merge_2Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2*
_output_shapes

:@: *
T0*
N
l
'coarse/coarse/conv3-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv3-bn/ExpandDims
ExpandDims'coarse/coarse/conv3-bn/ExpandDims/input%coarse/coarse/conv3-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv3-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv3-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv3-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv3-bn/ExpandDims_1/input'coarse/coarse/conv3-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv3-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
´
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Š
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
Ř
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
Ň
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ä
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
ł
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
ŕ
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
Ú
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
đ
(coarse/coarse/conv3-bn/AssignMovingAvg_1	AssignSubcoarse/conv3-bn/moving_variance,coarse/coarse/conv3-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking( *
T0*
_output_shapes
:@
}
coarse/coarse/conv3-reluRelu!coarse/coarse/conv3-bn/cond/Merge*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
Č
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Í
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @      *
_output_shapes
:
¸
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ź
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
ź
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
Ş
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
×
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 

%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
´
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
˛
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
ż
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
¨
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:

3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
î
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
T0*
data_formatNHWC

&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*  ?*
_output_shapes	
:
­
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
Ţ
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:

&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
Ť
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
Ű
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
Ź
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueB*    *
_output_shapes	
:
š
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
÷
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:

 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
ł
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueB*  ?*
_output_shapes	
:
Á
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 

&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ť
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
m
"coarse/coarse/conv4-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv4-bn/cond/switch_tIdentity$coarse/coarse/conv4-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv4-bn/cond/switch_fIdentity"coarse/coarse/conv4-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv4-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ź
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
Ő
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
::
Ó
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
::
ş
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::*
T0*
is_training(*
data_formatNHWC
ž
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
×
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
::
Ő
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
::
ă
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
::
ë
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
::
â
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::*
T0*
is_training( *
data_formatNHWC
Ę
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: *
T0*
N
ť
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
_output_shapes
	:: *
T0*
N
ť
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
_output_shapes
	:: *
T0*
N
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv4-bn/ExpandDims
ExpandDims'coarse/coarse/conv4-bn/ExpandDims/input%coarse/coarse/conv4-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv4-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv4-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv4-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv4-bn/ExpandDims_1/input'coarse/coarse/conv4-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv4-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
´
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Ş
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
Ů
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
Ó
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
ĺ
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:
´
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
á
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
Ű
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
ń
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
É
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
l
coarse/coarse/Reshape/shapeConst*
dtype0*
valueB"˙˙˙˙   *
_output_shapes
:

coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_2coarse/coarse/Reshape/shape*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
Tshape0
Š
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB"      *
_output_shapes
:

2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 

4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  ?*
_output_shapes
: 
ü
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
ţ
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
ě
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
Ż
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
Ü
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 

"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
Ł
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
Ë
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:
Ž
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Š
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"      *
_output_shapes
:

2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 

4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  ?*
_output_shapes
: 
ú
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
ü
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
ę
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
Ť
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
Ú
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	

"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
Ą
coarse/fc2/fc2-b
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
Ę
coarse/fc2/fc2-b/AssignAssigncoarse/fc2/fc2-b"coarse/fc2/fc2-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
}
coarse/fc2/fc2-b/readIdentitycoarse/fc2/fc2-b*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
ş
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
Pow/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
H
PowPowsubPow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
add/yConst*
dtype0*
valueB
 *Ěź+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
SqrtSqrtadd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
dtype0*
valueB"       *
_output_shapes
:
W
MeanMeanSqrtConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
N
	loss/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
Î
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
Đ
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
î
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
Ů
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ú
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 

gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
Ć
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0

gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
Ę
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0

gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
˛
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
°
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ă
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
×
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
´
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  ?*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ý
gradients/Pow_grad/Greater/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ă
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
Ú
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
ö
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ű
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
¸
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ă
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0*
data_formatNHWC
´
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
Š
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
ů
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
Ż
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
Á
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	
Đ
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
_output_shapes	
:*
T0*
data_formatNHWC
Î
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
Ý
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:
ú
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ô
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
: 
Ż
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
Â
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ŕ
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
: 
ý
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
ó
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ż
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ĺ
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
ˇ
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $

Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
Ő
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
Ů
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
ó
gradients/zeros_like	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:

Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::*
T0*
is_training( *
data_formatNHWC
Ą
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:
ó
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:

Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%o:*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙ $::: : *
T0*
is_training(*
data_formatNHWC

Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Ë
gradients/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
Ö
gradients/zeros/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: *
T0*
N

gradients/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_1/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes	
:
ó
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
_output_shapes
	:: *
T0*
N

gradients/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_2/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes	
:
ó
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
_output_shapes
	:: *
T0*
N
Í
gradients/Switch_3Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: *
T0*
N

gradients/Switch_4Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_4/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes	
:
ď
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
_output_shapes
	:: *
T0*
N

gradients/Switch_5Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_5/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes	
:
ď
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
_output_shapes
	:: *
T0*
N
Ő
gradients/AddNAddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
N
Ž
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
_output_shapes	
:*
T0*
data_formatNHWC
Á
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
ń
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
¨
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:
Č
gradients/AddN_1AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:*
N
Č
gradients/AddN_2AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:*
N
Ń
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ę
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
â
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
ş
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ $@
ś
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
Ý
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ä
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
ľ
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@

Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
Ô
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
Ř
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
ô
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ô
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ő
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ő
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@

Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
Ą
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
ó
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@

Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%o:*C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙@H@:@:@: : *
T0*
is_training(*
data_formatNHWC

Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Ë
gradients/Switch_6Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_6/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: *
T0*
N

gradients/Switch_7Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_8Shapegradients/Switch_7:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_7/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:@
ň
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
_output_shapes

:@: *
T0*
N

gradients/Switch_8Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_8/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes
:@
ň
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
_output_shapes

:@: *
T0*
N
Ë
gradients/Switch_9Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
˙
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: *
T0*
N

gradients/Switch_10Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_11Shapegradients/Switch_10*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_10/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes
:@
ď
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
_output_shapes

:@: *
T0*
N

gradients/Switch_11Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_12Shapegradients/Switch_11*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_11/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*
_output_shapes
:@
ď
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
_output_shapes

:@: *
T0*
N
Ö
gradients/AddN_3AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
N
Ż
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes
:@*
T0*
data_formatNHWC
Ă
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
ň
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
§
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
Ç
gradients/AddN_4AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@*
N
Ç
gradients/AddN_5AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@*
N
Ď
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ę
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ŕ
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
ş
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H 
ľ
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
Ű
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ä
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
š
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
Ö
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ú
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ő
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 

Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
T0*
is_training( *
data_formatNHWC
Ą
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
ó
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 

Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%o:*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ : : : : *
T0*
is_training(*
data_formatNHWC

Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Đ
gradients/Switch_12Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : *
T0*
N

gradients/Switch_13Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_14Shapegradients/Switch_13:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_13/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
: 
ó
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
_output_shapes

: : *
T0*
N

gradients/Switch_14Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_14/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
: 
ó
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
_output_shapes

: : *
T0*
N
Đ
gradients/Switch_15Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : *
T0*
N

gradients/Switch_16Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_17Shapegradients/Switch_16*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_16/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
: 
ď
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
_output_shapes

: : *
T0*
N

gradients/Switch_17Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_18Shapegradients/Switch_17*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_17/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
: 
ď
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
_output_shapes

: : *
T0*
N
Ř
gradients/AddN_6AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
N
Ż
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
_output_shapes
: *
T0*
data_formatNHWC
Ă
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
ô
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
§
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
Ç
gradients/AddN_7AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: *
N
Ç
gradients/AddN_8AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: *
N
Í
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ę
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ţ
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
ź
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
ĺ
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
Ŕ
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
Đ
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
¤
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ł
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ą
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ý
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter

Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:

beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 

beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ˇ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
s
beta1_power/readIdentitybeta1_power*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 

beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *wž?*
_output_shapes
: 

beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ˇ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
s
beta2_power/readIdentitybeta2_power*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
š
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
Ć
coarse/conv1/conv1-w/Adam
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ő
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
ť
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
Č
coarse/conv1/conv1-w/Adam_1
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ű
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Ł
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Ą
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
Ž
coarse/conv1/conv1-b/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
é
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
Ł
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
°
coarse/conv1/conv1-b/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ď
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
Í
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
Ú
#coarse/conv2-conv/conv2-conv-w/Adam
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 

*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
˝
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Ď
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
Ü
%coarse/conv2-conv/conv2-conv-w/Adam_1
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 
Ł
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
Á
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
ľ
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
Â
#coarse/conv2-conv/conv2-conv-b/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
ą
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
ˇ
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
Ä
%coarse/conv2-conv/conv2-conv-b/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
ľ
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
Ł
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
°
coarse/conv2-bn/gamma/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
í
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ľ
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
˛
coarse/conv2-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
ó
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ą
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
Ž
coarse/conv2-bn/beta/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
é
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Ł
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
°
coarse/conv2-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
ď
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Í
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
Ú
#coarse/conv3-conv/conv3-conv-w/Adam
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 

*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
˝
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Ď
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
Ü
%coarse/conv3-conv/conv3-conv-w/Adam_1
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 
Ł
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
Á
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
ľ
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
Â
#coarse/conv3-conv/conv3-conv-b/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
ą
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
ˇ
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
Ä
%coarse/conv3-conv/conv3-conv-b/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
ľ
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
Ł
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
°
coarse/conv3-bn/gamma/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
í
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ľ
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
˛
coarse/conv3-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
ó
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ą
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
Ž
coarse/conv3-bn/beta/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
é
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ł
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
°
coarse/conv3-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
ď
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ď
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@*    *'
_output_shapes
:@
Ü
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 

*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
ž
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
Ń
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@*    *'
_output_shapes
:@
Ţ
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
¤
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
Â
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
ˇ
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
Ä
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
˛
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:
š
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
Ć
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
ś
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:
Ľ
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*    *
_output_shapes	
:
˛
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
î
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:
§
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*    *
_output_shapes	
:
´
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
ô
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:
Ł
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
°
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
ę
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
Ľ
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
˛
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
đ
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
§
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB *    *!
_output_shapes
: 
´
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ŕ
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
Š
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB *    *!
_output_shapes
: 
ś
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ć
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 

'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
¨
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
Ú
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:

coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:

)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
Ş
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
ŕ
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:

coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:
Ł
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	*    *
_output_shapes
:	
°
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
Ţ
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
Ľ
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	*    *
_output_shapes
:	
˛
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
ä
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	

'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
Ś
coarse/fc2/fc2-b/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
Ů
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:

coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:

)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
¨
coarse/fc2/fc2-b/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
ß
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:

coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Ë

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Ë

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *wž?*
_output_shapes
: 
Í
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
¤
*Adam/update_coarse/conv1/conv1-w/ApplyAdam	ApplyAdamcoarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonNgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-w*
use_locking( *
T0*&
_output_shapes
:

*Adam/update_coarse/conv1/conv1-b/ApplyAdam	ApplyAdamcoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonRgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
:
ĺ
4Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam	ApplyAdamcoarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking( *
T0*&
_output_shapes
: 
Ý
4Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam	ApplyAdamcoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking( *
T0*
_output_shapes
: 
ß
+Adam/update_coarse/conv2-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_nesterov( *(
_class
loc:@coarse/conv2-bn/gamma*
use_locking( *
T0*
_output_shapes
: 
Ú
*Adam/update_coarse/conv2-bn/beta/ApplyAdam	ApplyAdamcoarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *'
_class
loc:@coarse/conv2-bn/beta*
use_locking( *
T0*
_output_shapes
: 
ĺ
4Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam	ApplyAdamcoarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking( *
T0*&
_output_shapes
: @
Ý
4Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam	ApplyAdamcoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking( *
T0*
_output_shapes
:@
ß
+Adam/update_coarse/conv3-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_nesterov( *(
_class
loc:@coarse/conv3-bn/gamma*
use_locking( *
T0*
_output_shapes
:@
Ú
*Adam/update_coarse/conv3-bn/beta/ApplyAdam	ApplyAdamcoarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( *'
_class
loc:@coarse/conv3-bn/beta*
use_locking( *
T0*
_output_shapes
:@
ć
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@
Ţ
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:
ŕ
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:
Ű
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:

&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
: 

&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:

&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	

&Adam/update_coarse/fc2/fc2-b/ApplyAdam	ApplyAdamcoarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-b*
use_locking( *
T0*
_output_shapes
:
Đ
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ň

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
Ł
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
˙
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
ż
save/SaveV2/tensor_namesConst*
dtype0*ň
valuečBĺ>Bbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:>
â
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:>

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powercoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1coarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1coarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1coarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1coarse/conv2-bn/moving_meancoarse/conv2-bn/moving_variancecoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1coarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1coarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1coarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1coarse/conv3-bn/moving_meancoarse/conv3-bn/moving_variancecoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1coarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1coarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1coarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1coarse/conv4-bn/moving_meancoarse/conv4-bn/moving_variancecoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1coarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1coarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1coarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1coarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1coarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1*L
dtypesB
@2>
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBbeta1_power*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst*
dtype0* 
valueBBbeta2_power*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
z
save/RestoreV2_2/tensor_namesConst*
dtype0*)
value BBcoarse/conv1/conv1-b*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_2Assigncoarse/conv1/conv1-bsave/RestoreV2_2*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv1/conv1-b/Adam*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_4/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv1/conv1-b/Adam_1*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
˝
save/Assign_4Assigncoarse/conv1/conv1-b/Adam_1save/RestoreV2_4*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
z
save/RestoreV2_5/tensor_namesConst*
dtype0*)
value BBcoarse/conv1/conv1-w*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Â
save/Assign_5Assigncoarse/conv1/conv1-wsave/RestoreV2_5*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

save/RestoreV2_6/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv1/conv1-w/Adam*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ç
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

save/RestoreV2_7/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv1/conv1-w/Adam_1*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
É
save/Assign_7Assigncoarse/conv1/conv1-w/Adam_1save/RestoreV2_7*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
z
save/RestoreV2_8/tensor_namesConst*
dtype0*)
value BBcoarse/conv2-bn/beta*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_8Assigncoarse/conv2-bn/betasave/RestoreV2_8*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_9/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv2-bn/beta/Adam*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_10/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv2-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_10Assigncoarse/conv2-bn/beta/Adam_1save/RestoreV2_10*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
|
save/RestoreV2_11/tensor_namesConst*
dtype0**
value!BBcoarse/conv2-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_12/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv2-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_13/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv2-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_14/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv2-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_15/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv2-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_16/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv2-conv/conv2-conv-b*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_17/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv2-conv/conv2-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_18/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv2-conv/conv2-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_19/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv2-conv/conv2-conv-w*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Ř
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 

save/RestoreV2_20/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv2-conv/conv2-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 

save/RestoreV2_21/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv2-conv/conv2-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_21Assign%coarse/conv2-conv/conv2-conv-w/Adam_1save/RestoreV2_21*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
{
save/RestoreV2_22/tensor_namesConst*
dtype0*)
value BBcoarse/conv3-bn/beta*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_23/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv3-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
˝
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_24/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv3-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_24Assigncoarse/conv3-bn/beta/Adam_1save/RestoreV2_24*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
|
save/RestoreV2_25/tensor_namesConst*
dtype0**
value!BBcoarse/conv3-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_26/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv3-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_27/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv3-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_28/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv3-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_29/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv3-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_30/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv3-conv/conv3-conv-b*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_31/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv3-conv/conv3-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_32/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv3-conv/conv3-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_33/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv3-conv/conv3-conv-w*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
Ř
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @

save/RestoreV2_34/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv3-conv/conv3-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @

save/RestoreV2_35/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv3-conv/conv3-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_35Assign%coarse/conv3-conv/conv3-conv-w/Adam_1save/RestoreV2_35*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
{
save/RestoreV2_36/tensor_namesConst*
dtype0*)
value BBcoarse/conv4-bn/beta*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
š
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_37/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv4-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_38/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv4-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
Ŕ
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:
|
save/RestoreV2_39/tensor_namesConst*
dtype0**
value!BBcoarse/conv4-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_40/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv4-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
Ŕ
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_41/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv4-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
Â
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_42/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv4-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ç
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_43/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv4-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_44/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv4-conv/conv4-conv-b*
_output_shapes
:
k
"save/RestoreV2_44/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
Í
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_45/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv4-conv/conv4-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
Ň
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_46/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv4-conv/conv4-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_46/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
Ô
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_47/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv4-conv/conv4-conv-w*
_output_shapes
:
k
"save/RestoreV2_47/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@

save/RestoreV2_48/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv4-conv/conv4-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_48/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ţ
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@

save/RestoreV2_49/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv4-conv/conv4-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_49/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
ŕ
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
w
save/RestoreV2_50/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-b*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_50Assigncoarse/fc1/fc1-bsave/RestoreV2_50*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
|
save/RestoreV2_51/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-b/Adam*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_51Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_51*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
~
save/RestoreV2_52/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_52Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_52*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
w
save/RestoreV2_53/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-w*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_53Assigncoarse/fc1/fc1-wsave/RestoreV2_53*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
|
save/RestoreV2_54/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-w/Adam*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_54Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_54*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
~
save/RestoreV2_55/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save/Assign_55Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_55*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
w
save/RestoreV2_56/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-b*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
°
save/Assign_56Assigncoarse/fc2/fc2-bsave/RestoreV2_56*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
|
save/RestoreV2_57/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-b/Adam*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_57Assigncoarse/fc2/fc2-b/Adamsave/RestoreV2_57*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
~
save/RestoreV2_58/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_58Assigncoarse/fc2/fc2-b/Adam_1save/RestoreV2_58*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
w
save/RestoreV2_59/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-w*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_59Assigncoarse/fc2/fc2-wsave/RestoreV2_59*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
|
save/RestoreV2_60/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-w/Adam*
_output_shapes
:
k
"save/RestoreV2_60/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_60Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_60*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
~
save/RestoreV2_61/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_61/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_61Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_61*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
Ş
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"ĽŘ$´ţâ     Kfq	nÉkÖAJńĹ
Ő,ł,
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
ë
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignSub
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
Č
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
î
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
í
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%ˇŃ8"
data_formatstringNHWC"
is_trainingbool(
°
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%ˇŃ8"
data_formatstringNHWC"
is_trainingbool(
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
Ó
MaxPool

input"T
output"T"
Ttype0:
2
	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ë
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5
Pow
x"T
y"T
z"T"
Ttype:
	2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
,
Sqrt
x"T
y"T"
Ttype:	
2
9
SqrtGrad
y"T
dy"T
z"T"
Ttype:	
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
9
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514óý
z
imgPlaceholder*
dtype0*&
shape:˙˙˙˙˙˙˙˙˙*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
labelPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
is_trainingPlaceholder*
dtype0
*
shape:*
_output_shapes
:
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
G
lrPlaceholder*
dtype0*
shape:*
_output_shapes
:
š
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
¤
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
Ś
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  ?*
_output_shapes
: 

Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w

5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:

1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Á
coarse/conv1/conv1-w
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ń
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:

&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
Š
coarse/conv1/conv1-b
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
Ú
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
á
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ç
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*
data_formatNHWC*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
¸
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ť
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
ť
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Š
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Ő
coarse/conv2-conv/conv2-conv-w
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 

%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
ł
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
°
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
˝
coarse/conv2-conv/conv2-conv-b
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
§
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 

3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ď
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*
data_formatNHWC*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  ?*
_output_shapes
: 
Ť
coarse/conv2-bn/gamma
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
Ý
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 

&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
Š
coarse/conv2-bn/beta
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
Ú
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Ş
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
ˇ
coarse/conv2-bn/moving_mean
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *.
_class$
" loc:@coarse/conv2-bn/moving_mean*
shared_name 
ö
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 

 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ą
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  ?*
_output_shapes
: 
ż
coarse/conv2-bn/moving_variance
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
shared_name 

&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
Ş
$coarse/conv2-bn/moving_variance/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
m
"coarse/coarse/conv2-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv2-bn/cond/switch_tIdentity$coarse/coarse/conv2-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv2-bn/cond/switch_fIdentity"coarse/coarse/conv2-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv2-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ž
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ó
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
Ń
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
ˇ
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : 
Ŕ
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ő
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
Ó
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
á
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
é
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
ß
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : 
Ë
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 
ş
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

: : 
ş
#coarse/coarse/conv2-bn/cond/Merge_2Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes

: : 
l
'coarse/coarse/conv2-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv2-bn/ExpandDims
ExpandDims'coarse/coarse/conv2-bn/ExpandDims/input%coarse/coarse/conv2-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv2-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv2-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv2-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv2-bn/ExpandDims_1/input'coarse/coarse/conv2-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv2-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
´
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Š
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
Ř
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
Ň
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ä
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
ł
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
ŕ
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
Ú
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
đ
(coarse/coarse/conv2-bn/AssignMovingAvg_1	AssignSubcoarse/conv2-bn/moving_variance,coarse/coarse/conv2-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking( *
T0*
_output_shapes
: 

coarse/coarse/conv2-reluRelu!coarse/coarse/conv2-bn/cond/Merge*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ć
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Í
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
¸
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ť
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
ť
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Š
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Ő
coarse/conv3-conv/conv3-conv-w
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 

%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
ł
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
°
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
˝
coarse/conv3-conv/conv3-conv-b
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
§
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@

3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
í
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*
data_formatNHWC*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  ?*
_output_shapes
:@
Ť
coarse/conv3-bn/gamma
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
Ý
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@

&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
Š
coarse/conv3-bn/beta
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
Ú
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ş
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
ˇ
coarse/conv3-bn/moving_mean
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
shared_name 
ö
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@

 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ą
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  ?*
_output_shapes
:@
ż
coarse/conv3-bn/moving_variance
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
shared_name 

&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
Ş
$coarse/conv3-bn/moving_variance/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
m
"coarse/coarse/conv3-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv3-bn/cond/switch_tIdentity$coarse/coarse/conv3-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv3-bn/cond/switch_fIdentity"coarse/coarse/conv3-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv3-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ş
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
Ó
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
Ń
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
ľ
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@
ź
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
Ő
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
Ó
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
á
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
é
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
Ý
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@
É
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: 
ş
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

:@: 
ş
#coarse/coarse/conv3-bn/cond/Merge_2Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes

:@: 
l
'coarse/coarse/conv3-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv3-bn/ExpandDims
ExpandDims'coarse/coarse/conv3-bn/ExpandDims/input%coarse/coarse/conv3-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv3-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv3-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv3-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv3-bn/ExpandDims_1/input'coarse/coarse/conv3-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv3-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
´
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Š
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
Ř
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
Ň
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ä
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
ł
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
ŕ
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
Ú
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
đ
(coarse/coarse/conv3-bn/AssignMovingAvg_1	AssignSubcoarse/conv3-bn/moving_variance,coarse/coarse/conv3-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking( *
T0*
_output_shapes
:@
}
coarse/coarse/conv3-reluRelu!coarse/coarse/conv3-bn/cond/Merge*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
Č
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Í
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @      *
_output_shapes
:
¸
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
ş
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  ?*
_output_shapes
: 
Ź
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
ź
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
Ş
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
×
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 

%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
´
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
˛
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
ż
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
¨
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:

3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
î
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*
data_formatNHWC*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*  ?*
_output_shapes	
:
­
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
Ţ
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:

&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
Ť
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
Ű
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
Ź
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueB*    *
_output_shapes	
:
š
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
÷
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:

 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
ł
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueB*  ?*
_output_shapes	
:
Á
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 

&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ť
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
m
"coarse/coarse/conv4-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv4-bn/cond/switch_tIdentity$coarse/coarse/conv4-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv4-bn/cond/switch_fIdentity"coarse/coarse/conv4-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv4-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:

!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 

#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
ź
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
Ő
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
::
Ó
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
::
ş
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::
ž
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
×
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
::
Ő
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
::
ă
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
::
ë
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
::
â
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::
Ę
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: 
ť
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:: 
ť
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes
	:: 
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
°
!coarse/coarse/conv4-bn/ExpandDims
ExpandDims'coarse/coarse/conv4-bn/ExpandDims/input%coarse/coarse/conv4-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv4-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv4-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
ś
#coarse/coarse/conv4-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv4-bn/ExpandDims_1/input'coarse/coarse/conv4-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv4-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
´
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:

coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
Ş
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
Ů
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
Ó
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:
ĺ
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:
´
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
á
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
Ű
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:
ń
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
É
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
l
coarse/coarse/Reshape/shapeConst*
dtype0*
valueB"˙˙˙˙   *
_output_shapes
:

coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_2coarse/coarse/Reshape/shape*
Tshape0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Š
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB"      *
_output_shapes
:

2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 

4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  ?*
_output_shapes
: 
ü
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
ţ
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
ě
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
Ż
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
Ü
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 

"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
Ł
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
Ë
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:
Ž
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"      *
_output_shapes
:

2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 

4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  ?*
_output_shapes
: 
ú
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
ü
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
ę
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
Ť
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
Ú
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	

"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
Ą
coarse/fc2/fc2-b
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
Ę
coarse/fc2/fc2-b/AssignAssigncoarse/fc2/fc2-b"coarse/fc2/fc2-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
}
coarse/fc2/fc2-b/readIdentitycoarse/fc2/fc2-b*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
ş
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
Pow/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
H
PowPowsubPow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
add/yConst*
dtype0*
valueB
 *Ěź+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
SqrtSqrtadd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
dtype0*
valueB"       *
_output_shapes
:
W
MeanMeanSqrtConst*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
N
	loss/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
Î
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
Đ
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
î
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
Ů
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ú
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 

gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
Ć
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
Ę
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
˛
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
°
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
ă
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
×
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
´
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  ?*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
gradients/Pow_grad/Greater/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
ă
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
Ú
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
ö
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ű
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:
´
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
Š
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
ů
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
Ż
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
Á
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	
Đ
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes	
:
Î
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
Ý
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:
ú
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ô
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
: 
Ż
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
Â
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ŕ
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
: 
ý
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
ó
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ĺ
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
ˇ
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $

Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
Ő
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
Ů
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
ó
gradients/zeros_like	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ő
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:

Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $::::
Ą
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:
ó
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:
ó
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:

Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙ $::: : 

Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Ë
gradients/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
Ö
gradients/zeros/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: 

gradients/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_1/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes	
:
ó
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
N*
T0*
_output_shapes
	:: 

gradients/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_2/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes	
:
ó
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
N*
T0*
_output_shapes
	:: 
Í
gradients/Switch_3Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙ $:˙˙˙˙˙˙˙˙˙ $
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $

Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ $: 

gradients/Switch_4Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_4/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes	
:
ď
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
N*
T0*
_output_shapes
	:: 

gradients/Switch_5Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
::
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_5/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes	
:
ď
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
N*
T0*
_output_shapes
	:: 
Ő
gradients/AddNAddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
Ž
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
T0*
_output_shapes	
:
Á
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
ń
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ $
¨
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:
Č
gradients/AddN_1AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:
Č
gradients/AddN_2AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:
Ń
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ę
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
â
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
ş
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ $@
ś
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@
Ý
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ä
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
ľ
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@

Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
Ô
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
Ř
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
ô
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ô
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ő
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ő
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@

Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙@H@:@:@:@:@
Ą
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
ó
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
ó
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@

Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙@H@:@:@: : 

Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Ë
gradients/Switch_6Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_6/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@

Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: 

gradients/Switch_7Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_8Shapegradients/Switch_7:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_7/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:@
ň
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
N*
T0*
_output_shapes

:@: 

gradients/Switch_8Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_8/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes
:@
ň
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
N*
T0*
_output_shapes

:@: 
Ë
gradients/Switch_9Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙@H@:˙˙˙˙˙˙˙˙˙@H@
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
Ř
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
˙
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@: 

gradients/Switch_10Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_11Shapegradients/Switch_10*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_10/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes
:@
ď
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
N*
T0*
_output_shapes

:@: 

gradients/Switch_11Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_12Shapegradients/Switch_11*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_11/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*
_output_shapes
:@
ď
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
N*
T0*
_output_shapes

:@: 
Ö
gradients/AddN_3AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
Ż
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
T0*
_output_shapes
:@
Ă
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
ň
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H@
§
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
Ç
gradients/AddN_4AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@
Ç
gradients/AddN_5AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@
Ď
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ę
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ŕ
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
ş
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@H 
ľ
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
Ű
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
Ä
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
š
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
Ö
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ú
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ő
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ő
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 

Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
data_formatNHWC*
T0*
is_training( *I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : 
Ą
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
ó
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
ó
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 

Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%o:*
data_formatNHWC*
T0*
is_training(*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ : : : : 

Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 

Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Đ
gradients/Switch_12Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 

gradients/Switch_13Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_14Shapegradients/Switch_13:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_13/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
: 
ó
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
N*
T0*
_output_shapes

: : 

gradients/Switch_14Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_14/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
: 
ó
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
N*
T0*
_output_shapes

: : 
Đ
gradients/Switch_15Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 

gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 

gradients/Switch_16Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_17Shapegradients/Switch_16*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_16/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
: 
ď
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
N*
T0*
_output_shapes

: : 

gradients/Switch_17Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_18Shapegradients/Switch_17*
out_type0*
T0*
_output_shapes
:
Ů
gradients/zeros_17/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
: 
ď
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
N*
T0*
_output_shapes

: : 
Ř
gradients/AddN_6AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ż
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
data_formatNHWC*
T0*
_output_shapes
: 
Ă
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
ô
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
§
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
Ç
gradients/AddN_7AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: 
Ç
gradients/AddN_8AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: 
Í
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ę
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ţ
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
ź
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
ĺ
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
Ŕ
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
Đ
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
¤
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ł
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ą
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ý
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter

Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:

beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 

beta1_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ˇ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
s
beta1_power/readIdentitybeta1_power*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 

beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *wž?*
_output_shapes
: 

beta2_power
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ˇ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
s
beta2_power/readIdentitybeta2_power*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
š
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
Ć
coarse/conv1/conv1-w/Adam
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ő
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
ť
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
Č
coarse/conv1/conv1-w/Adam_1
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ű
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Ł
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Ą
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
Ž
coarse/conv1/conv1-b/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
é
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
Ł
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
°
coarse/conv1/conv1-b/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-b*
shared_name 
ď
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
Í
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
Ú
#coarse/conv2-conv/conv2-conv-w/Adam
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 

*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
˝
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
Ď
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
Ü
%coarse/conv2-conv/conv2-conv-w/Adam_1
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
shared_name 
Ł
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
Á
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
ľ
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
Â
#coarse/conv2-conv/conv2-conv-b/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
ą
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
ˇ
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
Ä
%coarse/conv2-conv/conv2-conv-b/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
shared_name 

,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
ľ
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
Ł
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
°
coarse/conv2-bn/gamma/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
í
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ľ
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
˛
coarse/conv2-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *(
_class
loc:@coarse/conv2-bn/gamma*
shared_name 
ó
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ą
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
Ž
coarse/conv2-bn/beta/Adam
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
é
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Ł
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
°
coarse/conv2-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *'
_class
loc:@coarse/conv2-bn/beta*
shared_name 
ď
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
Í
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
Ú
#coarse/conv3-conv/conv3-conv-w/Adam
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 

*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
˝
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
Ď
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
Ü
%coarse/conv3-conv/conv3-conv-w/Adam_1
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
shared_name 
Ł
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
Á
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
ľ
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
Â
#coarse/conv3-conv/conv3-conv-b/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
ą
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
ˇ
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
Ä
%coarse/conv3-conv/conv3-conv-b/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
shared_name 

,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
ľ
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
Ł
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
°
coarse/conv3-bn/gamma/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
í
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ľ
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
˛
coarse/conv3-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*(
_class
loc:@coarse/conv3-bn/gamma*
shared_name 
ó
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ą
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
Ž
coarse/conv3-bn/beta/Adam
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
é
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ł
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
°
coarse/conv3-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*'
_class
loc:@coarse/conv3-bn/beta*
shared_name 
ď
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
Ď
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@*    *'
_output_shapes
:@
Ü
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 

*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
ž
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
Ń
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@*    *'
_output_shapes
:@
Ţ
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@*
dtype0*
shape:@*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
¤
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
Â
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@
ˇ
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
Ä
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
˛
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:
š
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB*    *
_output_shapes	
:
Ć
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 

,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:
ś
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:
Ľ
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*    *
_output_shapes	
:
˛
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
î
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:
§
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB*    *
_output_shapes	
:
´
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
ô
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:
Ł
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
°
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
ę
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
Ľ
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB*    *
_output_shapes	
:
˛
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
đ
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:
§
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB *    *!
_output_shapes
: 
´
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ŕ
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 
Š
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB *    *!
_output_shapes
: 
ś
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
: *
dtype0*
shape: *#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ć
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 

coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
: 

'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
¨
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
Ú
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:

coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:

)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB*    *
_output_shapes	
:
Ş
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
ŕ
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:

coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:
Ł
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	*    *
_output_shapes
:	
°
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
Ţ
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	
Ľ
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	*    *
_output_shapes
:	
˛
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	*
dtype0*
shape:	*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
ä
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	

coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	

'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
Ś
coarse/fc2/fc2-b/Adam
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
Ů
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:

coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:

)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
¨
coarse/fc2/fc2-b/Adam_1
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*#
_class
loc:@coarse/fc2/fc2-b*
shared_name 
ß
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:

coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Ë

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Ë

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *wž?*
_output_shapes
: 
Í
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
¤
*Adam/update_coarse/conv1/conv1-w/ApplyAdam	ApplyAdamcoarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonNgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-w*
use_locking( *
T0*&
_output_shapes
:

*Adam/update_coarse/conv1/conv1-b/ApplyAdam	ApplyAdamcoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonRgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
:
ĺ
4Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam	ApplyAdamcoarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking( *
T0*&
_output_shapes
: 
Ý
4Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam	ApplyAdamcoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking( *
T0*
_output_shapes
: 
ß
+Adam/update_coarse/conv2-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_nesterov( *(
_class
loc:@coarse/conv2-bn/gamma*
use_locking( *
T0*
_output_shapes
: 
Ú
*Adam/update_coarse/conv2-bn/beta/ApplyAdam	ApplyAdamcoarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *'
_class
loc:@coarse/conv2-bn/beta*
use_locking( *
T0*
_output_shapes
: 
ĺ
4Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam	ApplyAdamcoarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking( *
T0*&
_output_shapes
: @
Ý
4Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam	ApplyAdamcoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking( *
T0*
_output_shapes
:@
ß
+Adam/update_coarse/conv3-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_nesterov( *(
_class
loc:@coarse/conv3-bn/gamma*
use_locking( *
T0*
_output_shapes
:@
Ú
*Adam/update_coarse/conv3-bn/beta/ApplyAdam	ApplyAdamcoarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( *'
_class
loc:@coarse/conv3-bn/beta*
use_locking( *
T0*
_output_shapes
:@
ć
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@
Ţ
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:
ŕ
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:
Ű
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:

&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
: 

&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:

&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	

&Adam/update_coarse/fc2/fc2-b/ApplyAdam	ApplyAdamcoarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-b*
use_locking( *
T0*
_output_shapes
:
Đ
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ň

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
Ł
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
˙
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
ż
save/SaveV2/tensor_namesConst*
dtype0*ň
valuečBĺ>Bbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:>
â
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:>

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powercoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1coarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1coarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1coarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1coarse/conv2-bn/moving_meancoarse/conv2-bn/moving_variancecoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1coarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1coarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1coarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1coarse/conv3-bn/moving_meancoarse/conv3-bn/moving_variancecoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1coarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1coarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1coarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1coarse/conv4-bn/moving_meancoarse/conv4-bn/moving_variancecoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1coarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1coarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1coarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1coarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1coarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1*L
dtypesB
@2>
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBbeta1_power*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst*
dtype0* 
valueBBbeta2_power*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
: 
z
save/RestoreV2_2/tensor_namesConst*
dtype0*)
value BBcoarse/conv1/conv1-b*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_2Assigncoarse/conv1/conv1-bsave/RestoreV2_2*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv1/conv1-b/Adam*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_4/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv1/conv1-b/Adam_1*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
˝
save/Assign_4Assigncoarse/conv1/conv1-b/Adam_1save/RestoreV2_4*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
z
save/RestoreV2_5/tensor_namesConst*
dtype0*)
value BBcoarse/conv1/conv1-w*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Â
save/Assign_5Assigncoarse/conv1/conv1-wsave/RestoreV2_5*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

save/RestoreV2_6/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv1/conv1-w/Adam*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ç
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:

save/RestoreV2_7/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv1/conv1-w/Adam_1*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
É
save/Assign_7Assigncoarse/conv1/conv1-w/Adam_1save/RestoreV2_7*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
z
save/RestoreV2_8/tensor_namesConst*
dtype0*)
value BBcoarse/conv2-bn/beta*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_8Assigncoarse/conv2-bn/betasave/RestoreV2_8*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_9/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv2-bn/beta/Adam*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_10/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv2-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_10Assigncoarse/conv2-bn/beta/Adam_1save/RestoreV2_10*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
|
save/RestoreV2_11/tensor_namesConst*
dtype0**
value!BBcoarse/conv2-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_12/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv2-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_13/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv2-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_14/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv2-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_15/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv2-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_16/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv2-conv/conv2-conv-b*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_17/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv2-conv/conv2-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_18/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv2-conv/conv2-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_19/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv2-conv/conv2-conv-w*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Ř
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 

save/RestoreV2_20/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv2-conv/conv2-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 

save/RestoreV2_21/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv2-conv/conv2-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_21Assign%coarse/conv2-conv/conv2-conv-w/Adam_1save/RestoreV2_21*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
{
save/RestoreV2_22/tensor_namesConst*
dtype0*)
value BBcoarse/conv3-bn/beta*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_23/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv3-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
˝
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_24/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv3-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_24Assigncoarse/conv3-bn/beta/Adam_1save/RestoreV2_24*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
|
save/RestoreV2_25/tensor_namesConst*
dtype0**
value!BBcoarse/conv3-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_26/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv3-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_27/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv3-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
Á
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_28/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv3-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Ć
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_29/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv3-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_30/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv3-conv/conv3-conv-b*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ě
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_31/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv3-conv/conv3-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ń
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_32/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv3-conv/conv3-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
Ó
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_33/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv3-conv/conv3-conv-w*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
Ř
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @

save/RestoreV2_34/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv3-conv/conv3-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @

save/RestoreV2_35/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv3-conv/conv3-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_35Assign%coarse/conv3-conv/conv3-conv-w/Adam_1save/RestoreV2_35*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
{
save/RestoreV2_36/tensor_namesConst*
dtype0*)
value BBcoarse/conv4-bn/beta*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
š
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_37/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv4-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_38/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv4-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
Ŕ
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:
|
save/RestoreV2_39/tensor_namesConst*
dtype0**
value!BBcoarse/conv4-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ť
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_40/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv4-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
Ŕ
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_41/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv4-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
Â
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_42/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv4-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ç
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_43/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv4-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_44/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv4-conv/conv4-conv-b*
_output_shapes
:
k
"save/RestoreV2_44/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
Í
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_45/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv4-conv/conv4-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
Ň
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_46/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv4-conv/conv4-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_46/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
Ô
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_47/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv4-conv/conv4-conv-w*
_output_shapes
:
k
"save/RestoreV2_47/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@

save/RestoreV2_48/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv4-conv/conv4-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_48/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ţ
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@

save/RestoreV2_49/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv4-conv/conv4-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_49/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
ŕ
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@
w
save/RestoreV2_50/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-b*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_50Assigncoarse/fc1/fc1-bsave/RestoreV2_50*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
|
save/RestoreV2_51/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-b/Adam*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save/Assign_51Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_51*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
~
save/RestoreV2_52/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_52Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_52*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:
w
save/RestoreV2_53/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-w*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_53Assigncoarse/fc1/fc1-wsave/RestoreV2_53*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
|
save/RestoreV2_54/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-w/Adam*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_54Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_54*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
~
save/RestoreV2_55/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save/Assign_55Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_55*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
: 
w
save/RestoreV2_56/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-b*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
°
save/Assign_56Assigncoarse/fc2/fc2-bsave/RestoreV2_56*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
|
save/RestoreV2_57/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-b/Adam*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_57Assigncoarse/fc2/fc2-b/Adamsave/RestoreV2_57*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
~
save/RestoreV2_58/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_58Assigncoarse/fc2/fc2-b/Adam_1save/RestoreV2_58*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
w
save/RestoreV2_59/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-w*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_59Assigncoarse/fc2/fc2-wsave/RestoreV2_59*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
|
save/RestoreV2_60/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-w/Adam*
_output_shapes
:
k
"save/RestoreV2_60/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_60Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_60*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
~
save/RestoreV2_61/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_61/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_61Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_61*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	
Ş
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""ź:
cond_contextŤ:¨:
´
%coarse/coarse/conv2-bn/cond/cond_text%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_t:0 *š
#coarse/coarse/conv2-bn/cond/Const:0
%coarse/coarse/conv2-bn/cond/Const_1:0
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:1
5coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1
5coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1
,coarse/coarse/conv2-bn/cond/FusedBatchNorm:0
,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1
,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2
,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3
,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4
%coarse/coarse/conv2-bn/cond/pred_id:0
&coarse/coarse/conv2-bn/cond/switch_t:0
9coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add:0
coarse/conv2-bn/beta/read:0
coarse/conv2-bn/gamma/read:0T
coarse/conv2-bn/beta/read:05coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1U
coarse/conv2-bn/gamma/read:05coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1p
9coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add:03coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:1
ţ

'coarse/coarse/conv2-bn/cond/cond_text_1%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_f:0*

5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch:0
7coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1:0
7coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2:0
7coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3:0
7coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4:0
.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:0
.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1
.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2
.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3
.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4
%coarse/coarse/conv2-bn/cond/pred_id:0
&coarse/coarse/conv2-bn/cond/switch_f:0
9coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add:0
coarse/conv2-bn/beta/read:0
coarse/conv2-bn/gamma/read:0
"coarse/conv2-bn/moving_mean/read:0
&coarse/conv2-bn/moving_variance/read:0V
coarse/conv2-bn/beta/read:07coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2:0]
"coarse/conv2-bn/moving_mean/read:07coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3:0a
&coarse/conv2-bn/moving_variance/read:07coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4:0W
coarse/conv2-bn/gamma/read:07coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1:0r
9coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add:05coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch:0
´
%coarse/coarse/conv3-bn/cond/cond_text%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_t:0 *š
#coarse/coarse/conv3-bn/cond/Const:0
%coarse/coarse/conv3-bn/cond/Const_1:0
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:1
5coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1
5coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1
,coarse/coarse/conv3-bn/cond/FusedBatchNorm:0
,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1
,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2
,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3
,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4
%coarse/coarse/conv3-bn/cond/pred_id:0
&coarse/coarse/conv3-bn/cond/switch_t:0
9coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add:0
coarse/conv3-bn/beta/read:0
coarse/conv3-bn/gamma/read:0U
coarse/conv3-bn/gamma/read:05coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1p
9coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add:03coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:1T
coarse/conv3-bn/beta/read:05coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1
ţ

'coarse/coarse/conv3-bn/cond/cond_text_1%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_f:0*

5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch:0
7coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1:0
7coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2:0
7coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3:0
7coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4:0
.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:0
.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1
.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2
.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3
.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4
%coarse/coarse/conv3-bn/cond/pred_id:0
&coarse/coarse/conv3-bn/cond/switch_f:0
9coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add:0
coarse/conv3-bn/beta/read:0
coarse/conv3-bn/gamma/read:0
"coarse/conv3-bn/moving_mean/read:0
&coarse/conv3-bn/moving_variance/read:0a
&coarse/conv3-bn/moving_variance/read:07coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4:0W
coarse/conv3-bn/gamma/read:07coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1:0r
9coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add:05coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch:0V
coarse/conv3-bn/beta/read:07coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2:0]
"coarse/conv3-bn/moving_mean/read:07coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3:0
´
%coarse/coarse/conv4-bn/cond/cond_text%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_t:0 *š
#coarse/coarse/conv4-bn/cond/Const:0
%coarse/coarse/conv4-bn/cond/Const_1:0
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:1
5coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1
5coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1
,coarse/coarse/conv4-bn/cond/FusedBatchNorm:0
,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1
,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2
,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3
,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4
%coarse/coarse/conv4-bn/cond/pred_id:0
&coarse/coarse/conv4-bn/cond/switch_t:0
9coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add:0
coarse/conv4-bn/beta/read:0
coarse/conv4-bn/gamma/read:0T
coarse/conv4-bn/beta/read:05coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1p
9coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add:03coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:1U
coarse/conv4-bn/gamma/read:05coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1
ţ

'coarse/coarse/conv4-bn/cond/cond_text_1%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_f:0*

5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch:0
7coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1:0
7coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2:0
7coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3:0
7coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4:0
.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:0
.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1
.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2
.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3
.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4
%coarse/coarse/conv4-bn/cond/pred_id:0
&coarse/coarse/conv4-bn/cond/switch_f:0
9coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add:0
coarse/conv4-bn/beta/read:0
coarse/conv4-bn/gamma/read:0
"coarse/conv4-bn/moving_mean/read:0
&coarse/conv4-bn/moving_variance/read:0V
coarse/conv4-bn/beta/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2:0r
9coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add:05coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch:0W
coarse/conv4-bn/gamma/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1:0]
"coarse/conv4-bn/moving_mean/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3:0a
&coarse/conv4-bn/moving_variance/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4:0"ů
trainable_variablesáŢ

coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
Ż
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0
Ż
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0
Ż
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv4-conv/conv4-conv-b:0%coarse/conv4-conv/conv4-conv-b/Assign%coarse/conv4-conv/conv4-conv-b/read:022coarse/conv4-conv/conv4-conv-b/Initializer/Const:0

coarse/conv4-bn/gamma:0coarse/conv4-bn/gamma/Assigncoarse/conv4-bn/gamma/read:02(coarse/conv4-bn/gamma/Initializer/ones:0
|
coarse/conv4-bn/beta:0coarse/conv4-bn/beta/Assigncoarse/conv4-bn/beta/read:02(coarse/conv4-bn/beta/Initializer/zeros:0
w
coarse/fc1/fc1-w:0coarse/fc1/fc1-w/Assigncoarse/fc1/fc1-w/read:02/coarse/fc1/fc1-w/Initializer/truncated_normal:0
l
coarse/fc1/fc1-b:0coarse/fc1/fc1-b/Assigncoarse/fc1/fc1-b/read:02$coarse/fc1/fc1-b/Initializer/Const:0
w
coarse/fc2/fc2-w:0coarse/fc2/fc2-w/Assigncoarse/fc2/fc2-w/read:02/coarse/fc2/fc2-w/Initializer/truncated_normal:0
l
coarse/fc2/fc2-b:0coarse/fc2/fc2-b/Assigncoarse/fc2/fc2-b/read:02$coarse/fc2/fc2-b/Initializer/Const:0"J
	variablesJJ

coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
Ż
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0

coarse/conv2-bn/moving_mean:0"coarse/conv2-bn/moving_mean/Assign"coarse/conv2-bn/moving_mean/read:02/coarse/conv2-bn/moving_mean/Initializer/zeros:0
§
!coarse/conv2-bn/moving_variance:0&coarse/conv2-bn/moving_variance/Assign&coarse/conv2-bn/moving_variance/read:022coarse/conv2-bn/moving_variance/Initializer/ones:0
Ż
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0

coarse/conv3-bn/moving_mean:0"coarse/conv3-bn/moving_mean/Assign"coarse/conv3-bn/moving_mean/read:02/coarse/conv3-bn/moving_mean/Initializer/zeros:0
§
!coarse/conv3-bn/moving_variance:0&coarse/conv3-bn/moving_variance/Assign&coarse/conv3-bn/moving_variance/read:022coarse/conv3-bn/moving_variance/Initializer/ones:0
Ż
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
¤
 coarse/conv4-conv/conv4-conv-b:0%coarse/conv4-conv/conv4-conv-b/Assign%coarse/conv4-conv/conv4-conv-b/read:022coarse/conv4-conv/conv4-conv-b/Initializer/Const:0

coarse/conv4-bn/gamma:0coarse/conv4-bn/gamma/Assigncoarse/conv4-bn/gamma/read:02(coarse/conv4-bn/gamma/Initializer/ones:0
|
coarse/conv4-bn/beta:0coarse/conv4-bn/beta/Assigncoarse/conv4-bn/beta/read:02(coarse/conv4-bn/beta/Initializer/zeros:0

coarse/conv4-bn/moving_mean:0"coarse/conv4-bn/moving_mean/Assign"coarse/conv4-bn/moving_mean/read:02/coarse/conv4-bn/moving_mean/Initializer/zeros:0
§
!coarse/conv4-bn/moving_variance:0&coarse/conv4-bn/moving_variance/Assign&coarse/conv4-bn/moving_variance/read:022coarse/conv4-bn/moving_variance/Initializer/ones:0
w
coarse/fc1/fc1-w:0coarse/fc1/fc1-w/Assigncoarse/fc1/fc1-w/read:02/coarse/fc1/fc1-w/Initializer/truncated_normal:0
l
coarse/fc1/fc1-b:0coarse/fc1/fc1-b/Assigncoarse/fc1/fc1-b/read:02$coarse/fc1/fc1-b/Initializer/Const:0
w
coarse/fc2/fc2-w:0coarse/fc2/fc2-w/Assigncoarse/fc2/fc2-w/read:02/coarse/fc2/fc2-w/Initializer/truncated_normal:0
l
coarse/fc2/fc2-b:0coarse/fc2/fc2-b/Assigncoarse/fc2/fc2-b/read:02$coarse/fc2/fc2-b/Initializer/Const:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

coarse/conv1/conv1-w/Adam:0 coarse/conv1/conv1-w/Adam/Assign coarse/conv1/conv1-w/Adam/read:02-coarse/conv1/conv1-w/Adam/Initializer/zeros:0

coarse/conv1/conv1-w/Adam_1:0"coarse/conv1/conv1-w/Adam_1/Assign"coarse/conv1/conv1-w/Adam_1/read:02/coarse/conv1/conv1-w/Adam_1/Initializer/zeros:0

coarse/conv1/conv1-b/Adam:0 coarse/conv1/conv1-b/Adam/Assign coarse/conv1/conv1-b/Adam/read:02-coarse/conv1/conv1-b/Adam/Initializer/zeros:0

coarse/conv1/conv1-b/Adam_1:0"coarse/conv1/conv1-b/Adam_1/Assign"coarse/conv1/conv1-b/Adam_1/read:02/coarse/conv1/conv1-b/Adam_1/Initializer/zeros:0
¸
%coarse/conv2-conv/conv2-conv-w/Adam:0*coarse/conv2-conv/conv2-conv-w/Adam/Assign*coarse/conv2-conv/conv2-conv-w/Adam/read:027coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros:0
Ŕ
'coarse/conv2-conv/conv2-conv-w/Adam_1:0,coarse/conv2-conv/conv2-conv-w/Adam_1/Assign,coarse/conv2-conv/conv2-conv-w/Adam_1/read:029coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros:0
¸
%coarse/conv2-conv/conv2-conv-b/Adam:0*coarse/conv2-conv/conv2-conv-b/Adam/Assign*coarse/conv2-conv/conv2-conv-b/Adam/read:027coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros:0
Ŕ
'coarse/conv2-conv/conv2-conv-b/Adam_1:0,coarse/conv2-conv/conv2-conv-b/Adam_1/Assign,coarse/conv2-conv/conv2-conv-b/Adam_1/read:029coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros:0

coarse/conv2-bn/gamma/Adam:0!coarse/conv2-bn/gamma/Adam/Assign!coarse/conv2-bn/gamma/Adam/read:02.coarse/conv2-bn/gamma/Adam/Initializer/zeros:0

coarse/conv2-bn/gamma/Adam_1:0#coarse/conv2-bn/gamma/Adam_1/Assign#coarse/conv2-bn/gamma/Adam_1/read:020coarse/conv2-bn/gamma/Adam_1/Initializer/zeros:0

coarse/conv2-bn/beta/Adam:0 coarse/conv2-bn/beta/Adam/Assign coarse/conv2-bn/beta/Adam/read:02-coarse/conv2-bn/beta/Adam/Initializer/zeros:0

coarse/conv2-bn/beta/Adam_1:0"coarse/conv2-bn/beta/Adam_1/Assign"coarse/conv2-bn/beta/Adam_1/read:02/coarse/conv2-bn/beta/Adam_1/Initializer/zeros:0
¸
%coarse/conv3-conv/conv3-conv-w/Adam:0*coarse/conv3-conv/conv3-conv-w/Adam/Assign*coarse/conv3-conv/conv3-conv-w/Adam/read:027coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros:0
Ŕ
'coarse/conv3-conv/conv3-conv-w/Adam_1:0,coarse/conv3-conv/conv3-conv-w/Adam_1/Assign,coarse/conv3-conv/conv3-conv-w/Adam_1/read:029coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros:0
¸
%coarse/conv3-conv/conv3-conv-b/Adam:0*coarse/conv3-conv/conv3-conv-b/Adam/Assign*coarse/conv3-conv/conv3-conv-b/Adam/read:027coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros:0
Ŕ
'coarse/conv3-conv/conv3-conv-b/Adam_1:0,coarse/conv3-conv/conv3-conv-b/Adam_1/Assign,coarse/conv3-conv/conv3-conv-b/Adam_1/read:029coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros:0

coarse/conv3-bn/gamma/Adam:0!coarse/conv3-bn/gamma/Adam/Assign!coarse/conv3-bn/gamma/Adam/read:02.coarse/conv3-bn/gamma/Adam/Initializer/zeros:0

coarse/conv3-bn/gamma/Adam_1:0#coarse/conv3-bn/gamma/Adam_1/Assign#coarse/conv3-bn/gamma/Adam_1/read:020coarse/conv3-bn/gamma/Adam_1/Initializer/zeros:0

coarse/conv3-bn/beta/Adam:0 coarse/conv3-bn/beta/Adam/Assign coarse/conv3-bn/beta/Adam/read:02-coarse/conv3-bn/beta/Adam/Initializer/zeros:0

coarse/conv3-bn/beta/Adam_1:0"coarse/conv3-bn/beta/Adam_1/Assign"coarse/conv3-bn/beta/Adam_1/read:02/coarse/conv3-bn/beta/Adam_1/Initializer/zeros:0
¸
%coarse/conv4-conv/conv4-conv-w/Adam:0*coarse/conv4-conv/conv4-conv-w/Adam/Assign*coarse/conv4-conv/conv4-conv-w/Adam/read:027coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros:0
Ŕ
'coarse/conv4-conv/conv4-conv-w/Adam_1:0,coarse/conv4-conv/conv4-conv-w/Adam_1/Assign,coarse/conv4-conv/conv4-conv-w/Adam_1/read:029coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros:0
¸
%coarse/conv4-conv/conv4-conv-b/Adam:0*coarse/conv4-conv/conv4-conv-b/Adam/Assign*coarse/conv4-conv/conv4-conv-b/Adam/read:027coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros:0
Ŕ
'coarse/conv4-conv/conv4-conv-b/Adam_1:0,coarse/conv4-conv/conv4-conv-b/Adam_1/Assign,coarse/conv4-conv/conv4-conv-b/Adam_1/read:029coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros:0

coarse/conv4-bn/gamma/Adam:0!coarse/conv4-bn/gamma/Adam/Assign!coarse/conv4-bn/gamma/Adam/read:02.coarse/conv4-bn/gamma/Adam/Initializer/zeros:0

coarse/conv4-bn/gamma/Adam_1:0#coarse/conv4-bn/gamma/Adam_1/Assign#coarse/conv4-bn/gamma/Adam_1/read:020coarse/conv4-bn/gamma/Adam_1/Initializer/zeros:0

coarse/conv4-bn/beta/Adam:0 coarse/conv4-bn/beta/Adam/Assign coarse/conv4-bn/beta/Adam/read:02-coarse/conv4-bn/beta/Adam/Initializer/zeros:0

coarse/conv4-bn/beta/Adam_1:0"coarse/conv4-bn/beta/Adam_1/Assign"coarse/conv4-bn/beta/Adam_1/read:02/coarse/conv4-bn/beta/Adam_1/Initializer/zeros:0

coarse/fc1/fc1-w/Adam:0coarse/fc1/fc1-w/Adam/Assigncoarse/fc1/fc1-w/Adam/read:02)coarse/fc1/fc1-w/Adam/Initializer/zeros:0

coarse/fc1/fc1-w/Adam_1:0coarse/fc1/fc1-w/Adam_1/Assigncoarse/fc1/fc1-w/Adam_1/read:02+coarse/fc1/fc1-w/Adam_1/Initializer/zeros:0

coarse/fc1/fc1-b/Adam:0coarse/fc1/fc1-b/Adam/Assigncoarse/fc1/fc1-b/Adam/read:02)coarse/fc1/fc1-b/Adam/Initializer/zeros:0

coarse/fc1/fc1-b/Adam_1:0coarse/fc1/fc1-b/Adam_1/Assigncoarse/fc1/fc1-b/Adam_1/read:02+coarse/fc1/fc1-b/Adam_1/Initializer/zeros:0

coarse/fc2/fc2-w/Adam:0coarse/fc2/fc2-w/Adam/Assigncoarse/fc2/fc2-w/Adam/read:02)coarse/fc2/fc2-w/Adam/Initializer/zeros:0

coarse/fc2/fc2-w/Adam_1:0coarse/fc2/fc2-w/Adam_1/Assigncoarse/fc2/fc2-w/Adam_1/read:02+coarse/fc2/fc2-w/Adam_1/Initializer/zeros:0

coarse/fc2/fc2-b/Adam:0coarse/fc2/fc2-b/Adam/Assigncoarse/fc2/fc2-b/Adam/read:02)coarse/fc2/fc2-b/Adam/Initializer/zeros:0

coarse/fc2/fc2-b/Adam_1:0coarse/fc2/fc2-b/Adam_1/Assigncoarse/fc2/fc2-b/Adam_1/read:02+coarse/fc2/fc2-b/Adam_1/Initializer/zeros:0"
train_op

Adam"

update_ops

(coarse/coarse/conv2-bn/AssignMovingAvg:0
*coarse/coarse/conv2-bn/AssignMovingAvg_1:0
(coarse/coarse/conv3-bn/AssignMovingAvg:0
*coarse/coarse/conv3-bn/AssignMovingAvg_1:0
(coarse/coarse/conv4-bn/AssignMovingAvg:0
*coarse/coarse/conv4-bn/AssignMovingAvg_1:0"
	summaries


loss:0źÓ 