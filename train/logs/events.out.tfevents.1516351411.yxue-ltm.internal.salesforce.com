       �K"	  ��k��Abrain.Event:2����      �LY8	����k��A"��
z
imgPlaceholder*
dtype0*&
shape:�����������*1
_output_shapes
:�����������
h
labelPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
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
�
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
�
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
�
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  �?*
_output_shapes
: 
�
Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w
�
5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
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
�
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*1
_output_shapes
:�����������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*1
_output_shapes
:�����������*
T0*
data_formatNHWC
�
coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*1
_output_shapes
:�����������
�
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
�
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
�
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
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
�
%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*1
_output_shapes
:����������� *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*1
_output_shapes
:����������� *
T0*
data_formatNHWC
�
&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  �?*
_output_shapes
: 
�
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
�
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
�
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
�
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
�
 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  �?*
_output_shapes
: 
�
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
�
&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
�
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
�
!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::����������� :����������� 
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
�
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%o�:*I
_output_shapes7
5:����������� : : : : *
T0*
is_training(*
data_formatNHWC
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::����������� :����������� 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
�
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*I
_output_shapes7
5:����������� : : : : *
T0*
is_training( *
data_formatNHWC
�
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*3
_output_shapes!
:����������� : *
T0*
N
�
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
_output_shapes

: : *
T0*
N
�
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
�#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
�
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
�
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
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
:����������� 
�
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:���������@H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
�
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
�
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
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
�
%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:���������@H@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*/
_output_shapes
:���������@H@*
T0*
data_formatNHWC
�
&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  �?*
_output_shapes
:@
�
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
�
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
�
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
�
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
�
 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  �?*
_output_shapes
:@
�
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
�
&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
�
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
�
!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
�
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%o�:*G
_output_shapes5
3:���������@H@:@:@:@:@*
T0*
is_training(*
data_formatNHWC
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
�
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*G
_output_shapes5
3:���������@H@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
�
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*1
_output_shapes
:���������@H@: *
T0*
N
�
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
_output_shapes

:@: *
T0*
N
�
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
�#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
�
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
�
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
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
:���������@H@
�
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:��������� $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @   �   *
_output_shapes
:
�
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@�*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
�
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:��������� $�*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*0
_output_shapes
:��������� $�*
T0*
data_formatNHWC
�
&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*  �?*
_output_shapes	
:�
�
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
�
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:�
�
 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueB�*  �?*
_output_shapes	
:�
�
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 
�
&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:�
�
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
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
�
!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:�:�
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:�:�
�
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%o�:*L
_output_shapes:
8:��������� $�:�:�:�:�*
T0*
is_training(*
data_formatNHWC
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
:�:�
�
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*L
_output_shapes:
8:��������� $�:�:�:�:�*
T0*
is_training( *
data_formatNHWC
�
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*2
_output_shapes 
:��������� $�: *
T0*
N
�
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
_output_shapes
	:�: *
T0*
N
�
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
_output_shapes
	:�: *
T0*
N
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
�
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:�
�
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:�
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:��������� $�
�
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:����������*
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
valueB"���� �  *
_output_shapes
:
�
coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_2coarse/coarse/Reshape/shape*)
_output_shapes
:�����������*
T0*
Tshape0
�
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB" �  �   *
_output_shapes
:
�
2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 
�
4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  �?*
_output_shapes
: 
�
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
:���*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
�
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
�
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
�
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"�      *
_output_shapes
:
�
2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 
�
4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  �?*
_output_shapes
: 
�
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	�*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
�
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
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
�
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:���������
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
:���������
J
add/yConst*
dtype0*
valueB
 *̼�+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:���������
C
SqrtSqrtadd*
T0*'
_output_shapes
:���������
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
�
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0
�
gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0
�
gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
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
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  �?*
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
:���������
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
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
:���������
�
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
�
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
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
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0*
data_formatNHWC
�
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
�
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
�
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�
�
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
�
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:����������
�
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	�
�
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
�
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:����������
�
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:�
�
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:�����������
�
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:���
�
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
�
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:�����������
�
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
:���
�
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*0
_output_shapes
:����������*
T0*
Tshape0
�
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:��������� $�*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:��������� $�
�
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:��������� $�
�
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:��������� $�
�
gradients/zeros_like	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*L
_output_shapes:
8:��������� $�:�:�:�:�*
T0*
is_training( *
data_formatNHWC
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:��������� $�
�
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%o�:*F
_output_shapes4
2:��������� $�:�:�: : *
T0*
is_training(*
data_formatNHWC
�
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:��������� $�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:��������� $�:��������� $�
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
�
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
:��������� $�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*2
_output_shapes 
:��������� $�: *
T0*
N
�
gradients/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
_output_shapes
	:�: *
T0*
N
�
gradients/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
_output_shapes
	:�: *
T0*
N
�
gradients/Switch_3Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:��������� $�:��������� $�
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:��������� $�
�
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*2
_output_shapes 
:��������� $�: *
T0*
N
�
gradients/Switch_4Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
_output_shapes
	:�: *
T0*
N
�
gradients/Switch_5Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
_output_shapes
	:�: *
T0*
N
�
gradients/AddNAddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:��������� $�*
N
�
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:��������� $�
�
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:�
�
gradients/AddN_1AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:�*
N
�
gradients/AddN_2AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:�*
N
�
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
�
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:��������� $@
�
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@�
�
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:���������@H@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:���������@H@
�
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:���������@H@
�
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:���������@H@
�
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*G
_output_shapes5
3:���������@H@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:���������@H@
�
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%o�:*C
_output_shapes1
/:���������@H@:@:@: : *
T0*
is_training(*
data_formatNHWC
�
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:���������@H@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/Switch_6Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������@H@:���������@H@
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
�
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
:���������@H@
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*1
_output_shapes
:���������@H@: *
T0*
N
�
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
�
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
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
_output_shapes

:@: *
T0*
N
�
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
�
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
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
_output_shapes

:@: *
T0*
N
�
gradients/Switch_9Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������@H@:���������@H@
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������@H@
�
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*1
_output_shapes
:���������@H@: *
T0*
N
�
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
�
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
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
_output_shapes

:@: *
T0*
N
�
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
�
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
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
_output_shapes

:@: *
T0*
N
�
gradients/AddN_3AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:���������@H@*
N
�
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes
:@*
T0*
data_formatNHWC
�
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:���������@H@
�
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
�
gradients/AddN_4AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@*
N
�
gradients/AddN_5AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@*
N
�
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
�
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������@H 
�
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
�
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*1
_output_shapes
:����������� *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*1
_output_shapes
:����������� 
�
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*N
_output_shapes<
::����������� :����������� 
�
Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:����������� 
�
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:����������� 
�
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*I
_output_shapes7
5:����������� : : : : *
T0*
is_training( *
data_formatNHWC
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:����������� 
�
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%o�:*E
_output_shapes3
1:����������� : : : : *
T0*
is_training(*
data_formatNHWC
�
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:����������� 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/Switch_12Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::����������� :����������� 
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*1
_output_shapes
:����������� 
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*3
_output_shapes!
:����������� : *
T0*
N
�
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
�
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
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
_output_shapes

: : *
T0*
N
�
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
�
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
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
_output_shapes

: : *
T0*
N
�
gradients/Switch_15Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::����������� :����������� 
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*1
_output_shapes
:����������� 
�
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*3
_output_shapes!
:����������� : *
T0*
N
�
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
�
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
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
_output_shapes

: : *
T0*
N
�
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
�
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
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
_output_shapes

: : *
T0*
N
�
gradients/AddN_6AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:����������� *
N
�
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
_output_shapes
: *
T0*
data_formatNHWC
�
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:����������� 
�
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
gradients/AddN_7AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: *
N
�
gradients/AddN_8AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: *
N
�
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
�
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:�����������
�
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
�
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*1
_output_shapes
:�����������
�
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
�
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*1
_output_shapes
:�����������
�
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
�
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
�
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter
�
Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:�����������
�
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
�
beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 
�
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
�
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
�
beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *w�?*
_output_shapes
: 
�
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
�
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
�
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
�
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
�
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
�
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
�
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
�
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
�
*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
�
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
�
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
�
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
�
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
�
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
�
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
�
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
�
*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
�
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
�
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
�
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
�
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
�
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
�
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@�*    *'
_output_shapes
:@�
�
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@�*    *'
_output_shapes
:@�
�
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB���*    *!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB���*    *!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	�*    *
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	�*    *
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
�
)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
�

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
�

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w�?*
_output_shapes
: 
�
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@�
�
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:�
�
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:�
�
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:�
�
&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
:���
�
&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:�
�
&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	�
�
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
�
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�>Bbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:>
�
save/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:>
�
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
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
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
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
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
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
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
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
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
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
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
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
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
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
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
�
save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
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
�
save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
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
�
save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_50Assigncoarse/fc1/fc1-bsave/RestoreV2_50*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_51Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_51*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_52Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_52*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_53Assigncoarse/fc1/fc1-wsave/RestoreV2_53*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_54Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_54*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_55Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_55*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_59Assigncoarse/fc2/fc2-wsave/RestoreV2_59*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
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
�
save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_60Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_60*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
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
�
save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_61Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_61*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"�6<]��     Kfq	ס�k��AJ��
�,�,
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
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignSub
ref"T�

value"T

output_ref"T�"
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
�
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
�
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
�
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
�
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
epsilonfloat%��8"
data_formatstringNHWC"
is_trainingbool(
�
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
epsilonfloat%��8"
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
�
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
�
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
2	�
�
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
2	�
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
�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514��
z
imgPlaceholder*
dtype0*&
shape:�����������*1
_output_shapes
:�����������
h
labelPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
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
�
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
�
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
�
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  �?*
_output_shapes
: 
�
Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w
�
5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
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
�
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*1
_output_shapes
:�����������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*
data_formatNHWC*
T0*1
_output_shapes
:�����������
�
coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*1
_output_shapes
:�����������
�
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
�
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
�
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
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
�
%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*1
_output_shapes
:����������� *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*
data_formatNHWC*
T0*1
_output_shapes
:����������� 
�
&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  �?*
_output_shapes
: 
�
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
�
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
�
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
�
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
�
 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  �?*
_output_shapes
: 
�
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
�
&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
�
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
�
!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::����������� :����������� 
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
�
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*I
_output_shapes7
5:����������� : : : : 
�
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*N
_output_shapes<
::����������� :����������� 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
�
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
�
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *I
_output_shapes7
5:����������� : : : : 
�
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*
N*
T0*3
_output_shapes!
:����������� : 
�
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

: : 
�
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
�#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
�
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
�
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
�
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
�
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
:����������� 
�
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:���������@H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
�
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
�
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
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
�
%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:���������@H@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@H@
�
&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  �?*
_output_shapes
:@
�
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
�
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
�
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
�
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
�
 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  �?*
_output_shapes
:@
�
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
�
&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
�
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
�
!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
�
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*G
_output_shapes5
3:���������@H@:@:@:@:@
�
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
�
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
�
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:���������@H@:@:@:@:@
�
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*
N*
T0*1
_output_shapes
:���������@H@: 
�
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

:@: 
�
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
�#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
�
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
�
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
�
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
�
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
:���������@H@
�
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:��������� $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @   �   *
_output_shapes
:
�
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
�
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  �?*
_output_shapes
: 
�
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@�*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
�
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:��������� $�*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*
data_formatNHWC*
T0*0
_output_shapes
:��������� $�
�
&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*  �?*
_output_shapes	
:�
�
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
�
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:�
�
 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueB�*  �?*
_output_shapes	
:�
�
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 
�
&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:�
�
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
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
�
!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
�
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:�:�
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:�:�
�
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*L
_output_shapes:
8:��������� $�:�:�:�:�
�
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
:�:�
�
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
:�:�
�
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:��������� $�:�:�:�:�
�
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:��������� $�: 
�
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:�: 
�
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes
	:�: 
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
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
�
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
�
coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
�
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:
�
coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:�
�
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:�
�
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:�
�
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:�
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:��������� $�
�
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:����������*
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
valueB"���� �  *
_output_shapes
:
�
coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_2coarse/coarse/Reshape/shape*
Tshape0*
T0*)
_output_shapes
:�����������
�
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB" �  �   *
_output_shapes
:
�
2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 
�
4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  �?*
_output_shapes
: 
�
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
:���*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
�
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
�
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
�
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"�      *
_output_shapes
:
�
2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 
�
4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  �?*
_output_shapes
: 
�
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	�*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
�
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
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
�
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:���������
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
:���������
J
add/yConst*
dtype0*
valueB
 *̼�+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:���������
C
SqrtSqrtadd*
T0*'
_output_shapes
:���������
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
�
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
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
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
�
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  �?*
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
:���������
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
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
:���������
�
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
�
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
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
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:
�
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
�
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
�
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�
�
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
�
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:����������
�
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	�
�
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes	
:�
�
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
�
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:����������
�
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:�
�
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:�����������
�
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:���
�
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
�
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:�����������
�
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
:���
�
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
�
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:����������
�
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:��������� $�*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:��������� $�
�
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:��������� $�:��������� $�
�
Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:��������� $�
�
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:��������� $�
�
gradients/zeros_like	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:��������� $�:�:�:�:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:��������� $�
�
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*F
_output_shapes4
2:��������� $�:�:�: : 
�
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:��������� $�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:�
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:��������� $�:��������� $�
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
�
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
:��������� $�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*
T0*2
_output_shapes 
:��������� $�: 
�
gradients/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
N*
T0*
_output_shapes
	:�: 
�
gradients/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
N*
T0*
_output_shapes
	:�: 
�
gradients/Switch_3Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:��������� $�:��������� $�
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:��������� $�
�
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*2
_output_shapes 
:��������� $�: 
�
gradients/Switch_4Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
N*
T0*
_output_shapes
	:�: 
�
gradients/Switch_5Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:�:�
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
�
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
:�
�
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
N*
T0*
_output_shapes
	:�: 
�
gradients/AddNAddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:��������� $�
�
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
T0*
_output_shapes	
:�
�
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:��������� $�
�
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:�
�
gradients/AddN_1AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:�
�
gradients/AddN_2AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:�
�
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
�
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:��������� $@
�
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@�
�
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:���������@H@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:���������@H@
�
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:���������@H@:���������@H@
�
Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:���������@H@
�
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:���������@H@
�
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:���������@H@:@:@:@:@
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:���������@H@
�
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*C
_output_shapes1
/:���������@H@:@:@: : 
�
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:���������@H@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/Switch_6Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������@H@:���������@H@
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
�
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
:���������@H@
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*
T0*1
_output_shapes
:���������@H@: 
�
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
�
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
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
N*
T0*
_output_shapes

:@: 
�
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
�
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
�
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
N*
T0*
_output_shapes

:@: 
�
gradients/Switch_9Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������@H@:���������@H@
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������@H@
�
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
N*
T0*1
_output_shapes
:���������@H@: 
�
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
�
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
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
N*
T0*
_output_shapes

:@: 
�
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
�
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
�
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
N*
T0*
_output_shapes

:@: 
�
gradients/AddN_3AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:���������@H@
�
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
T0*
_output_shapes
:@
�
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:���������@H@
�
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
�
gradients/AddN_4AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@
�
gradients/AddN_5AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@
�
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
�
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������@H 
�
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
�
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*1
_output_shapes
:����������� *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*1
_output_shapes
:����������� 
�
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*N
_output_shapes<
::����������� :����������� 
�
Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
�
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:����������� 
�
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*1
_output_shapes
:����������� 
�
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training( *I
_output_shapes7
5:����������� : : : : 
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:����������� 
�
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%o�:*
data_formatNHWC*
T0*
is_training(*E
_output_shapes3
1:����������� : : : : 
�
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*1
_output_shapes
:����������� 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
�
gradients/Switch_12Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::����������� :����������� 
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*1
_output_shapes
:����������� 
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
N*
T0*3
_output_shapes!
:����������� : 
�
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
�
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
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
N*
T0*
_output_shapes

: : 
�
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
�
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
�
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
N*
T0*
_output_shapes

: : 
�
gradients/Switch_15Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
T0*N
_output_shapes<
::����������� :����������� 
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
�
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*1
_output_shapes
:����������� 
�
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
N*
T0*3
_output_shapes!
:����������� : 
�
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
�
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
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
N*
T0*
_output_shapes

: : 
�
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
�
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
�
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
N*
T0*
_output_shapes

: : 
�
gradients/AddN_6AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:����������� 
�
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
data_formatNHWC*
T0*
_output_shapes
: 
�
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
�
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*1
_output_shapes
:����������� 
�
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
gradients/AddN_7AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: 
�
gradients/AddN_8AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: 
�
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
�
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
�
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:�����������
�
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
�
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*1
_output_shapes
:�����������
�
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
�
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*1
_output_shapes
:�����������
�
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
�
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
�
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter
�
Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:�����������
�
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
�
beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 
�
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
�
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
�
beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *w�?*
_output_shapes
: 
�
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
�
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
�
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
�
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
�
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
�
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
�
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
�
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
�
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
�
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
�
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
�
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
�
*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
�
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
�
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
�
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
�
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
�
,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
�
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
�
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
�
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
�
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
�
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
�
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
�
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
�
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
�
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
�
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
�
*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
�
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
�
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
�
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
�
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
�
,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
�
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
�
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
�
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
�
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
�
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
�
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
�
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
�
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
�
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@�*    *'
_output_shapes
:@�
�
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@�*    *'
_output_shapes
:@�
�
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@�*
dtype0*
shape:@�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
�
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@�
�
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueB�*    *
_output_shapes	
:�
�
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
�
,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:�
�
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
�
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:�
�
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueB�*    *
_output_shapes	
:�
�
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
�
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:�
�
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB���*    *!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueB���*    *!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
:���*
dtype0*
shape:���*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
�
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
�
coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:���
�
'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueB�*    *
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
�
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
�
coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:�
�
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	�*    *
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	�*    *
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
�
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	�
�
'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
�
)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
�
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
�
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
�
coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
�

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
�

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w�?*
_output_shapes
: 
�
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@�
�
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:�
�
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:�
�
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:�
�
&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
:���
�
&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:�
�
&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	�
�
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
�
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�>Bbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:>
�
save/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:>
�
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
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
�
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
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
�
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
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
�
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
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
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
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
�
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
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
�
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
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
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
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
�
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
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:�
�
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
�
save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
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
�
save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
�
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
�
save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@�
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
�
save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_50Assigncoarse/fc1/fc1-bsave/RestoreV2_50*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_51Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_51*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_52Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_52*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:�
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
�
save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_53Assigncoarse/fc1/fc1-wsave/RestoreV2_53*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_54Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_54*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_55Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_55*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:���
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
�
save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
�
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
�
save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_59Assigncoarse/fc2/fc2-wsave/RestoreV2_59*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
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
�
save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_60Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_60*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
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
�
save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_61Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_61*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	�
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""�:
cond_context�:�:
�
%coarse/coarse/conv2-bn/cond/cond_text%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_t:0 *�
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
�

'coarse/coarse/conv2-bn/cond/cond_text_1%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_f:0*�

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
�
%coarse/coarse/conv3-bn/cond/cond_text%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_t:0 *�
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
�

'coarse/coarse/conv3-bn/cond/cond_text_1%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_f:0*�

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
�
%coarse/coarse/conv4-bn/cond/cond_text%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_t:0 *�
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
�

'coarse/coarse/conv4-bn/cond/cond_text_1%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_f:0*�

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
&coarse/conv4-bn/moving_variance/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4:0"�
trainable_variables��
�
coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
�
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
�
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0
�
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
�
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0
�
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
�
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
coarse/fc2/fc2-b:0coarse/fc2/fc2-b/Assigncoarse/fc2/fc2-b/read:02$coarse/fc2/fc2-b/Initializer/Const:0"�J
	variables�J�J
�
coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
�
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
�
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0
�
coarse/conv2-bn/moving_mean:0"coarse/conv2-bn/moving_mean/Assign"coarse/conv2-bn/moving_mean/read:02/coarse/conv2-bn/moving_mean/Initializer/zeros:0
�
!coarse/conv2-bn/moving_variance:0&coarse/conv2-bn/moving_variance/Assign&coarse/conv2-bn/moving_variance/read:022coarse/conv2-bn/moving_variance/Initializer/ones:0
�
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
�
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0
�
coarse/conv3-bn/moving_mean:0"coarse/conv3-bn/moving_mean/Assign"coarse/conv3-bn/moving_mean/read:02/coarse/conv3-bn/moving_mean/Initializer/zeros:0
�
!coarse/conv3-bn/moving_variance:0&coarse/conv3-bn/moving_variance/Assign&coarse/conv3-bn/moving_variance/read:022coarse/conv3-bn/moving_variance/Initializer/ones:0
�
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
�
 coarse/conv4-conv/conv4-conv-b:0%coarse/conv4-conv/conv4-conv-b/Assign%coarse/conv4-conv/conv4-conv-b/read:022coarse/conv4-conv/conv4-conv-b/Initializer/Const:0

coarse/conv4-bn/gamma:0coarse/conv4-bn/gamma/Assigncoarse/conv4-bn/gamma/read:02(coarse/conv4-bn/gamma/Initializer/ones:0
|
coarse/conv4-bn/beta:0coarse/conv4-bn/beta/Assigncoarse/conv4-bn/beta/read:02(coarse/conv4-bn/beta/Initializer/zeros:0
�
coarse/conv4-bn/moving_mean:0"coarse/conv4-bn/moving_mean/Assign"coarse/conv4-bn/moving_mean/read:02/coarse/conv4-bn/moving_mean/Initializer/zeros:0
�
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
�
coarse/conv1/conv1-w/Adam:0 coarse/conv1/conv1-w/Adam/Assign coarse/conv1/conv1-w/Adam/read:02-coarse/conv1/conv1-w/Adam/Initializer/zeros:0
�
coarse/conv1/conv1-w/Adam_1:0"coarse/conv1/conv1-w/Adam_1/Assign"coarse/conv1/conv1-w/Adam_1/read:02/coarse/conv1/conv1-w/Adam_1/Initializer/zeros:0
�
coarse/conv1/conv1-b/Adam:0 coarse/conv1/conv1-b/Adam/Assign coarse/conv1/conv1-b/Adam/read:02-coarse/conv1/conv1-b/Adam/Initializer/zeros:0
�
coarse/conv1/conv1-b/Adam_1:0"coarse/conv1/conv1-b/Adam_1/Assign"coarse/conv1/conv1-b/Adam_1/read:02/coarse/conv1/conv1-b/Adam_1/Initializer/zeros:0
�
%coarse/conv2-conv/conv2-conv-w/Adam:0*coarse/conv2-conv/conv2-conv-w/Adam/Assign*coarse/conv2-conv/conv2-conv-w/Adam/read:027coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros:0
�
'coarse/conv2-conv/conv2-conv-w/Adam_1:0,coarse/conv2-conv/conv2-conv-w/Adam_1/Assign,coarse/conv2-conv/conv2-conv-w/Adam_1/read:029coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros:0
�
%coarse/conv2-conv/conv2-conv-b/Adam:0*coarse/conv2-conv/conv2-conv-b/Adam/Assign*coarse/conv2-conv/conv2-conv-b/Adam/read:027coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros:0
�
'coarse/conv2-conv/conv2-conv-b/Adam_1:0,coarse/conv2-conv/conv2-conv-b/Adam_1/Assign,coarse/conv2-conv/conv2-conv-b/Adam_1/read:029coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros:0
�
coarse/conv2-bn/gamma/Adam:0!coarse/conv2-bn/gamma/Adam/Assign!coarse/conv2-bn/gamma/Adam/read:02.coarse/conv2-bn/gamma/Adam/Initializer/zeros:0
�
coarse/conv2-bn/gamma/Adam_1:0#coarse/conv2-bn/gamma/Adam_1/Assign#coarse/conv2-bn/gamma/Adam_1/read:020coarse/conv2-bn/gamma/Adam_1/Initializer/zeros:0
�
coarse/conv2-bn/beta/Adam:0 coarse/conv2-bn/beta/Adam/Assign coarse/conv2-bn/beta/Adam/read:02-coarse/conv2-bn/beta/Adam/Initializer/zeros:0
�
coarse/conv2-bn/beta/Adam_1:0"coarse/conv2-bn/beta/Adam_1/Assign"coarse/conv2-bn/beta/Adam_1/read:02/coarse/conv2-bn/beta/Adam_1/Initializer/zeros:0
�
%coarse/conv3-conv/conv3-conv-w/Adam:0*coarse/conv3-conv/conv3-conv-w/Adam/Assign*coarse/conv3-conv/conv3-conv-w/Adam/read:027coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros:0
�
'coarse/conv3-conv/conv3-conv-w/Adam_1:0,coarse/conv3-conv/conv3-conv-w/Adam_1/Assign,coarse/conv3-conv/conv3-conv-w/Adam_1/read:029coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros:0
�
%coarse/conv3-conv/conv3-conv-b/Adam:0*coarse/conv3-conv/conv3-conv-b/Adam/Assign*coarse/conv3-conv/conv3-conv-b/Adam/read:027coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros:0
�
'coarse/conv3-conv/conv3-conv-b/Adam_1:0,coarse/conv3-conv/conv3-conv-b/Adam_1/Assign,coarse/conv3-conv/conv3-conv-b/Adam_1/read:029coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros:0
�
coarse/conv3-bn/gamma/Adam:0!coarse/conv3-bn/gamma/Adam/Assign!coarse/conv3-bn/gamma/Adam/read:02.coarse/conv3-bn/gamma/Adam/Initializer/zeros:0
�
coarse/conv3-bn/gamma/Adam_1:0#coarse/conv3-bn/gamma/Adam_1/Assign#coarse/conv3-bn/gamma/Adam_1/read:020coarse/conv3-bn/gamma/Adam_1/Initializer/zeros:0
�
coarse/conv3-bn/beta/Adam:0 coarse/conv3-bn/beta/Adam/Assign coarse/conv3-bn/beta/Adam/read:02-coarse/conv3-bn/beta/Adam/Initializer/zeros:0
�
coarse/conv3-bn/beta/Adam_1:0"coarse/conv3-bn/beta/Adam_1/Assign"coarse/conv3-bn/beta/Adam_1/read:02/coarse/conv3-bn/beta/Adam_1/Initializer/zeros:0
�
%coarse/conv4-conv/conv4-conv-w/Adam:0*coarse/conv4-conv/conv4-conv-w/Adam/Assign*coarse/conv4-conv/conv4-conv-w/Adam/read:027coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros:0
�
'coarse/conv4-conv/conv4-conv-w/Adam_1:0,coarse/conv4-conv/conv4-conv-w/Adam_1/Assign,coarse/conv4-conv/conv4-conv-w/Adam_1/read:029coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros:0
�
%coarse/conv4-conv/conv4-conv-b/Adam:0*coarse/conv4-conv/conv4-conv-b/Adam/Assign*coarse/conv4-conv/conv4-conv-b/Adam/read:027coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros:0
�
'coarse/conv4-conv/conv4-conv-b/Adam_1:0,coarse/conv4-conv/conv4-conv-b/Adam_1/Assign,coarse/conv4-conv/conv4-conv-b/Adam_1/read:029coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros:0
�
coarse/conv4-bn/gamma/Adam:0!coarse/conv4-bn/gamma/Adam/Assign!coarse/conv4-bn/gamma/Adam/read:02.coarse/conv4-bn/gamma/Adam/Initializer/zeros:0
�
coarse/conv4-bn/gamma/Adam_1:0#coarse/conv4-bn/gamma/Adam_1/Assign#coarse/conv4-bn/gamma/Adam_1/read:020coarse/conv4-bn/gamma/Adam_1/Initializer/zeros:0
�
coarse/conv4-bn/beta/Adam:0 coarse/conv4-bn/beta/Adam/Assign coarse/conv4-bn/beta/Adam/read:02-coarse/conv4-bn/beta/Adam/Initializer/zeros:0
�
coarse/conv4-bn/beta/Adam_1:0"coarse/conv4-bn/beta/Adam_1/Assign"coarse/conv4-bn/beta/Adam_1/read:02/coarse/conv4-bn/beta/Adam_1/Initializer/zeros:0
�
coarse/fc1/fc1-w/Adam:0coarse/fc1/fc1-w/Adam/Assigncoarse/fc1/fc1-w/Adam/read:02)coarse/fc1/fc1-w/Adam/Initializer/zeros:0
�
coarse/fc1/fc1-w/Adam_1:0coarse/fc1/fc1-w/Adam_1/Assigncoarse/fc1/fc1-w/Adam_1/read:02+coarse/fc1/fc1-w/Adam_1/Initializer/zeros:0
�
coarse/fc1/fc1-b/Adam:0coarse/fc1/fc1-b/Adam/Assigncoarse/fc1/fc1-b/Adam/read:02)coarse/fc1/fc1-b/Adam/Initializer/zeros:0
�
coarse/fc1/fc1-b/Adam_1:0coarse/fc1/fc1-b/Adam_1/Assigncoarse/fc1/fc1-b/Adam_1/read:02+coarse/fc1/fc1-b/Adam_1/Initializer/zeros:0
�
coarse/fc2/fc2-w/Adam:0coarse/fc2/fc2-w/Adam/Assigncoarse/fc2/fc2-w/Adam/read:02)coarse/fc2/fc2-w/Adam/Initializer/zeros:0
�
coarse/fc2/fc2-w/Adam_1:0coarse/fc2/fc2-w/Adam_1/Assigncoarse/fc2/fc2-w/Adam_1/read:02+coarse/fc2/fc2-w/Adam_1/Initializer/zeros:0
�
coarse/fc2/fc2-b/Adam:0coarse/fc2/fc2-b/Adam/Assigncoarse/fc2/fc2-b/Adam/read:02)coarse/fc2/fc2-b/Adam/Initializer/zeros:0
�
coarse/fc2/fc2-b/Adam_1:0coarse/fc2/fc2-b/Adam_1/Assigncoarse/fc2/fc2-b/Adam_1/read:02+coarse/fc2/fc2-b/Adam_1/Initializer/zeros:0"
train_op

Adam"�

update_ops�
�
(coarse/coarse/conv2-bn/AssignMovingAvg:0
*coarse/coarse/conv2-bn/AssignMovingAvg_1:0
(coarse/coarse/conv3-bn/AssignMovingAvg:0
*coarse/coarse/conv3-bn/AssignMovingAvg_1:0
(coarse/coarse/conv4-bn/AssignMovingAvg:0
*coarse/coarse/conv4-bn/AssignMovingAvg_1:0"
	summaries


loss:0!]d�       ��-	M�k��A	*

loss�4Ek��       ��-	� �k��A*

loss�NE븕       ��-	�;Rl��A*

loss��8Dٽ\	       ��-	��l��A'*

loss��1D"P�q       ��-	�k5l��A1*

loss��DCߊ�       ��-	��#l��A;*

lossط�C�3��       ��-	j�S-l��AE*

loss^U�C]�ot       ��-	��v6l��AO*

loss�3�C�K��       ��-	���?l��AY*

loss��Cg�w       ��-	�?Hl��Ac*

loss�}�C��u       ��-	���Sl��Am*

loss�C?�$       ��-	�a�[l��Aw*

loss�^sC�n�j       �	n dl��A�*

loss�`�C:X��       �	̳�kl��A�*

loss?�C"�M�       �	�F�sl��A�*

lossr��B��E�       �	�j{l��A�*

loss��CC��	       �	�t�l��A�*

lossH��C���       �	 ���l��A�*

lossy��B�,       �	C;�l��A�*

lossLACjxF       �	��̙l��A�*

loss��B��B       �	�`��l��A�*

loss��B=*�       �	q��l��A�*

loss�'�B�	N       �	�U��l��A�*

loss�h�B* p=       �	��l��A�*

lossr(�BP8       �	S���l��A�*

lossE��B8�       �	���l��A�*

loss���B�e�       �	Mܝ�l��A�*

loss�qB�ϯ�       �	�(�l��A�*

loss(
uB��FH       �	����l��A�*

loss�^�B�'�       �	 �2�l��A�*

loss�nqB�D�       �	���l��A�*

loss�3<B�       �	 T�l��A�*

lossL�bBT�@�       �	�=�m��A�*

lossvM!B���a       �	�fT	m��A�*

loss0�1Bƀ�V       �	��m��A�*

loss1�Bz1�       �	4G]m��A�*

lossj.NBrca�       �	���m��A�*

loss�oB�\�       �	1DW'm��A�*

loss8�.Bi{�       �	�J�.m��A�*

loss�C�A��       �	�26m��A�*

loss#��A].��       �	Qf�@m��A�*

loss��AP�p       �	cGHm��A�*

loss���A�       �	e��Om��A�*

lossH��A�*�=       �	f�.Wm��A�*

loss(i�AO��       �	c*�^m��A�*

loss�ǯA���       �	��fm��A�*

loss��A|q��       �	K[�mm��A�*

loss2�A���:       �	���tm��A�*

loss5��A�U�P       �	��|m��A�*

loss�2�A��G)       �	t�m��A�*

lossxcA��L�       �	�O�m��A�*

loss�O�A��-       �	�m�m��A�*

lossli�A¤.�       �	����m��A�*

loss��Am±R       �	Y�J�m��A�*

loss �A�d�q       �	�P��m��A�*

lossݸLAE�.       �	�×�m��A�*

loss�h�A��M       �	�$�m��A�*

lossh�~A���       �	.���m��A�*

lossO9RA��       �	�Oi�m��A�*

loss��5A�2u       �	����m��A�*

lossRA}*s       �	2��m��A�*

loss��CA��        �	�_N�m��A�*

loss�[�Am�ƾ       �	����m��A�*

loss��fA�P+       �	�v�m��A�*

loss�yA��]       �	�*
�m��A�*

loss���@� �I       �	�n��A�*

lossP�3AC0�D       �	�Dn��A�*

loss�� A�&�C       �	�V�n��A�*

loss�W4AC.�       �	�!in��A�*

lossxi�@�i��       �	Z�	$n��A�*

loss�A��\       �	 ��.n��A�*

loss� A���       �	σO6n��A�*

loss��SA�]�       �	Ĳ�=n��A�*

loss�.&Af�X       �	��sEn��A�*

loss�PA~b^       �	��Ln��A�*

loss�[�@���       �	�=Un��A�*

lossn�iAD%�o       �	4I�]n��A�*

loss��A�Tp�       �	7fn��A�*

loss���@-�0�       �	XTnn��A�*

lossAC�@�V�\       �	emvn��A�*

loss�.A��C       �	{�n��A�*

lossH��@Mm�       �	���n��A�*

loss�@�3        �	�K��n��A�*

loss��A�N0       �	0�X�n��A�*

loss�@�\J�       �	���n��A�*

lossX(�@^�B�       �	-�ɧn��A�*

loss��@�K�A       �	muh�n��A�*

lossn��@`�f       �	*p�n��A�*

loss��@U��[       �	�Ϥ�n��A�*

loss�|�@���       �	��K�n��A�*

loss2�@�}$z       �	�(�n��A�*

loss���@��>�       �	����n��A�*

loss��A�8�J       �	���n��A�*

loss<�Aw�       �	M��n��A�*

lossf��@�H�X       �	#L��n��A�*

loss���@3�       �	K>[�n��A�*

loss1˝@}���       �	���n��A�*

loss�d�@���       �	qwo��A�*

loss�֢@4
,�       �	o��A�*

loss�$�@�       �	���o��A�*

loss���@>��       �	S^o��A�*

loss�d�@Y�iT       �	�a�"o��A�*

loss���@0��       �	%xs'o��A�*

loss���@D�X       �	z�+o��A�*

loss���@	���       �	|�0o��A�*

loss���@���#       �	�5o��A�*

lossh�@�}�       �	��9o��A�*

loss��@5���       �	� ,>o��A�*

loss}@p�r�       �	S�Bo��A�*

loss�EA�(�#       �	�5Go��A�*

lossޛ�@a�       �	��Mo��A�*

losss`�@��       �	��iRo��A�*

lossR��@����       �	�d�Vo��A�*

lossp�@y��J       �	dsl[o��A�*

loss�(�@�	r       �	"o�_o��A�*

loss���@ڧ'u       �	{Npdo��A�	*

loss(ck@��=       �	�T�ho��A�	*

loss�L�@ T>�       �	kkmo��A�	*

lossc$�@�l�{       �	[]�qo��A�	*

loss�{�@u�`�       �	��lvo��A�	*

loss.�|@rG�;       �	�C4}o��A�	*

loss�j�@�v��       �	E�o��A�	*

loss0[X@pE�       �	ץ�o��A�	*

loss��c@�"X;       �	�7��o��A�	*

loss47�@wJL�       �	�OS�o��A�	*

loss!	�@�*�9       �	����o��A�	*

lossD�m@�ϧ       �	�D��o��A�	*

loss���@�6�`       �	;9g�o��A�	*

loss�_�@��A�       �	���o��A�
*

loss.4@��B�       �	��o��A�
*

loss�mD@��e       �	󫩮o��A�
*

loss�zx@O�-{       �	�"c�o��A�
*

lossTI@O
�:       �	��1�o��A�
*

loss���@�dL       �	����o��A�
*

loss���@�kmK       �	I��o��A�
*

lossJ��@�5�<       �	i��o��A�
*

loss�&d@Hx��       �	�)F�o��A�
*

lossR��@M��       �	���o��A�
*

loss��@쪅       �	����o��A�
*

loss��?@.�%�       �	�p��o��A�
*

lossʋB@v)       �	`���o��A�*

lossxKN@���       �	q��o��A�*

loss���@H���       �	�O�o��A�*

lossc�u@�'
       �	���o��A�*

loss�)�@��2�       �	���o��A�*

loss˼@Zzu       �	8K��o��A�*

loss�kS@�c��       �	�*U�o��A�*

loss�|A@��5       �	��p��A�*

lossܘ�@�㕶       �	E,�p��A�*

loss�wG@�Yh%       �	3Q�p��A�*

loss��=@r�Ń       �	Q��p��A�*

loss8�@�,1       �	�mp��A�*

loss��@�:L       �	�L&p��A�*

loss���?
       �	o�� p��A�*

loss��P@1h^4       �	N��%p��A�*

loss��@�S�       �	*p��A�*

loss�i@~F�'       �	��\/p��A�*

loss�0@�f       �	��*4p��A�*

loss��b@Ud�H       �	��8p��A�*

loss��1@�:�       �	]4�=p��A�*

loss�x@`{�3       �	���Dp��A�*

loss��@ҍB       �	��GIp��A�*

loss;�O@�|�       �	�tNp��A�*

loss*	z@�g�r       �	ܞ�Rp��A�*

lossR�a@F}��       �	�*�Wp��A�*

loss+}�@��B�       �	}�h\p��A�*

lossv8�@d�W       �	*S$ap��A�*

loss��P@j(�       �	]��ep��A�*

loss��U@���       �	YN�jp��A�*

loss v\@��       �	�	lop��A�*

lossu�@>�l-       �	��)vp��A�*

loss�'@� B       �	�
�zp��A�*

loss���@���       �	�K�p��A�*

loss��@���       �	${h�p��A�*

lossR�@�9�Y       �	}?%�p��A�*

loss�CN@�       �	��ߍp��A�*

loss�&@�)       �	�Ţ�p��A�*

loss
�f@_��v       �	i6e�p��A�*

lossR۞@̄w�       �	a�&�p��A�*

loss�9�?)T=�       �	�Y�p��A�*

loss
�N@㖓�       �	ʉ��p��A�*

lossX@B@j�?       �	��l�p��A�*

losst�n@�\>k       �	��)�p��A�*

loss��S@RE	       �	.9�p��A�*

loss07@liZ*       �	dɹ�p��A�*

loss��@5���       �	2�s�p��A�*

lossm,@{/��       �	��@�p��A�*

lossN@��       �	; �p��A�*

loss�U@�Vs�       �	x`��p��A�*

lossr�%@�j�       �	좍�p��A�*

loss��@��       �	�'K�p��A�*

loss��g@���l       �	���p��A�*

loss@��?�       �	F���p��A�*

lossh��?���r       �	����p��A�*

loss�Ґ@��j#       �	��_�p��A�*

losst�	@�L       �	���p��A�*

loss>�@�       �	 F��p��A�*

loss��@ :�(       �	���p��A�*

loss@�A@X<@*       �	+ND�p��A�*

loss�k-@8��	       �	�q��A�*

loss^h+@#ƞ�       �	S��
q��A�*

loss�@���q       �	2�q��A�*

loss&�-@r�>h       �	��kq��A�*

loss]T�?!aV�       �	�-q��A�*

loss��?�s�       �	|e�q��A�*

loss6� @��s_       �	슴"q��A�*

lossu��@`�       �	�.l'q��A�*

loss��?č�	       �	a3,q��A�*

loss�&�?���       �	�J�0q��A�*

lossWP�?���s       �	�"�5q��A�*

loss/�X@�{�       �	
�<q��A�*

loss@��g       �	'zAq��A�*

loss�@ �e       �	�$1Fq��A�*

loss>�
@��;       �	N�Kq��A�*

loss%��?M�>       �	%�Oq��A�*

loss�@pL�        �	9|Tq��A�*

loss�)@��*�       �	� GYq��A�*

loss K�?��r�       �	~�]q��A�*

loss6�F@ƍ�       �	� �bq��A�*

loss�@]/�_       �	|zgq��A�*

loss��?���       �	y1nq��A�*

loss�2�@����       �	�<�rq��A�*

loss<��?5]�l       �	�C�wq��A�*

loss�$@Y��       �	��x|q��A�*

loss���?C�A�       �	��:�q��A�*

loss>|@�b       �	Qj��q��A�*

loss�#@�1��       �	����q��A�*

loss-��?�.�       �	��r�q��A�*

loss���?z6/       �	Ie)�q��A�*

loss�@��\�       �	���q��A�*

loss.�@[,W4       �	P��q��A�*

loss�r@���       �	홰�q��A�*

lossR��?� �l       �	�.i�q��A�*

loss^V�?c�3       �	6�"�q��A�*

loss�@@��gt       �	�]�q��A�*

lossj�=@��qw       �	=��q��A�*

loss^.@�[�       �	5�Y�q��A�*

loss�!�?�N��       �	���q��A�*

loss��(@�U�       �	Q.��q��A�*

loss�@38W�       �	6���q��A�*

lossB@	�8       �	̚w�q��A�*

loss�b@���       �	:?1�q��A�*

loss���?k)�       �	���q��A�*

loss>�p@���        �	�'��q��A�*

lossR��?0���       �	�z��q��A�*

losso��?,�@H       �	=}A�q��A�*

lossf@e("       �	����q��A�*

lossQ�@�       �	����q��A�*

loss�U[@=S�]       �	c{��q��A�*

loss��#@z�P�       �	��q��A�*

lossT@>1[�       �	N�r��A�*

lossfV�?���       �	�s�	r��A�*

loss��?�W       �	�8�r��A�*

lossw�?�_>       �	<l�r��A�*

loss|�H@!I�4       �	TS�r��A�*

loss
�@���       �	U�@r��A�*

loss.��?�W{^       �	�j�!r��A�*

lossg��?����       �	N*�&r��A�*

loss>b�?��>r       �	��p+r��A�*

loss���?�<��       �	��50r��A�*

losst��?Ke>        �	uG7r��A�*

lossT@����       �	��<r��A�*

lossT@�`��       �	b��@r��A�*

loss �@���       �	��Er��A�*

loss��7@IF��       �	��PJr��A�*

loss�.>@��d�       �	��Or��A�*

loss�+�?�ȈL       �	p��Sr��A�*

loss�@��       �	�ΔXr��A�*

loss��?o4�       �	��N]r��A�*

loss�e&@���       �	�|br��A�*

loss�@Á�z       �	hZ�hr��A�*

loss���?j��       �	Z.�mr��A�*

lossW��?/~"�       �	�Urr��A�*

loss��?���       �	Kwwr��A�*

loss��?Ge�?       �	���{r��A�*

lossb�?��|�       �	Nb��r��A�*

loss*h@��G       �	�mZ�r��A�*

loss�@e}��       �	��r��A�*

lossĢ@��0M       �	�͎r��A�*

loss�2�?d@s$       �	���r��A�*

loss�?��N]       �	:@�r��A�*

lossǉ
@��Ǵ       �	,���r��A�*

lossNy@�ԇ       �	�j��r��A�*

losss��?O NR       �	>˄�r��A�*

lossF�?���l       �	H4M�r��A�*

loss�;b?Tk=+       �	K;	�r��A�*

losssF�?�"�       �	+�Ŷr��A�*

loss��?b���       �	ٓ��r��A�*

loss�� @V�{�       �	8�G�r��A�*

loss���?���       �	&��r��A�*

loss��	@Xj�Z       �	 (�r��A�*

loss�8�?V"S       �	����r��A�*

loss(Z�?��l-       �	a2��r��A�*

lossL�/@ТK�       �	�F�r��A�*

loss��@R�N|       �	N��r��A�*

loss�@_���       �	�X��r��A�*

loss�S�?lWv       �	?���r��A�*

loss��:@���r       �	��I�r��A�*

loss�@�rp�       �	���r��A�*

loss�@��       �	 p��r��A�*

loss�U@�ʗ�       �	�B��r��A�*

loss���? ���       �	�%qs��A�*

loss�q]@D��       �	?.s��A�*

lossDH@D���       �	�+�s��A�*

loss�ؿ?U��
       �	z��s��A�*

loss8o'@��1n       �	ۅ]s��A�*

loss��@�E5�       �	��s��A�*

loss50@��       �	���s��A�*

loss�ɩ?K��       �	��#s��A�*

loss��?�f��       �	O<(s��A�*

loss�|@�c\       �	��W/s��A�*

loss�*�?EɯN       �	G9_4s��A�*

lossfI�?tU�       �	4�x9s��A�*

loss�$�?h��O       �	�ˊ>s��A�*

loss���?�57       �	xҪCs��A�*

lossI��?��       �	+��Hs��A�*

loss� �?s���       �	d�Ms��A�*

loss�-�?F��-       �	!��Rs��A�*

loss�M@�O��       �	��Ws��A�*

loss���?u3       �	}!�\s��A�*

loss�_�?�͑�       �	�~Wds��A�*

loss}��?�g�       �	��jis��A�*

lossf��?Ș��       �	e6vns��A�*

loss�@ND�A       �	��|ss��A�*

loss��@�VD�       �	{یxs��A�*

losse4�?�1�       �	ʋ�}s��A�*

loss���?�sl       �	�*��s��A�*

loss-N�?���       �	�x��s��A�*

loss��@E�H�       �	���s��A�*

loss�ܲ?rD       �	�!ڑs��A�*

loss�I�?���       �	���s��A�*

loss�ǭ?�$�       �	rM�s��A�*

loss��?cc��       �	�!�s��A�*

loss\�?��h�       �	q�=�s��A�*

loss%ڦ?M�E�       �	nL�s��A�*

loss�[�?<�N       �	��`�s��A�*

loss��?���       �	\so�s��A�*

loss���?.�Y       �	 &}�s��A�*

loss�C�?Ѽv�       �	��s��A�*

loss�Ci@yD�z       �	ӟ��s��A�*

losszs/@�-l       �	����s��A�*

lossL�?3��       �	��s��A�*

loss�s@��Zt       �	���s��A�*

losslb�?�	"�       �	Q�/�s��A�*

loss�M�?�8�       �	z�>�s��A�*

lossG�?,FX�       �	g�[�s��A�*

lossI��?��E�       �	�d}�s��A�*

loss��?�`�q       �	����s��A�*

loss���?�Lr+       �	�ڴ�s��A�*

loss���?Q�pH       �	���s��A�*

loss��?�0t       �	@��t��A�*

loss���?�ȸ       �	��t��A�*

loss��?g���       �	j%t��A�*

loss���?]�W�       �	f�=t��A�*

loss�$�?,���       �	6�Ot��A�*

loss���?�%O       �	<�bt��A�*

loss��?��N&       �	�z�!t��A�*

loss���?�F       �	�֤&t��A�*

loss�R�?��"X       �	���+t��A�*

loss���?��       �	4��0t��A�*

lossh��?�`l       �	5�8t��A�*

lossP��?͞,�       �	�'7=t��A�*

loss���?���!       �	\KBt��A�*

lossx҆?�^��       �	��bGt��A�*

lossl��?Y*8       �	�oLt��A�*

loss4��?颐�       �	�?}Qt��A�*

lossh�^?ښ6�       �	��Vt��A�*

lossTT_?�@>       �	$E�[t��A�*

loss*�@��q�       �	h\�`t��A�*

loss�R�?�ꓷ       �	�F�et��A�*

loss8v�?���c       �	���lt��A�*

loss+��?�v�       �	Y�rt��A�*

loss'��?�U�o       �	��wt��A�*

loss@��u       �	�-|t��A�*

loss1��?9r��       �	�?�t��A�*

lossԛ�?N��       �	�-a�t��A�*

loss�1�?.ҥ/       �	/���t��A�*

lossw�?�]1       �	2u��t��A�*

loss��?�C       �	SΜ�t��A�*

loss4�@c��       �	����t��A�*

loss��?���v       �	�7�t��A�*

loss���?*��       �	��t��A�*

loss���?��ǘ       �	��t��A�*

loss�w@V��       �	! 3�t��A�*

loss�6�?��k       �	x@�t��A�*

loss���?I�Z       �	��a�t��A�*

loss>��?m���       �	�.o�t��A�*

loss8��?��nL       �	��y�t��A�*

lossZ��?|o�A       �	͆�t��A�*

loss7��?��Y5       �	���t��A�*

loss�R�?Z<�       �	����t��A�*

loss�-�?���       �	ŏ��t��A�*

loss�7�?�Ns       �	����t��A�*

loss��t?���Q       �	���t��A�*

loss�%�?"٥�       �	aU�t��A�*

loss��@���       �	;+�t��A�*

loss/�E@�x�       �	�;�t��A�*

loss�@�[8       �	��F�t��A�*

lossR�@ ��N       �	}AU�t��A�*

lossX��?�f�       �	��au��A�*

loss���?��q�       �	zo�u��A�*

loss���?D��       �	�?�u��A�*

lossT��?��s       �	�u��A�*

loss-�&@Ŀx�       �	v6�u��A�*

loss�?]$\y       �	9+  u��A�*

loss�u�?(D��       �	M%u��A�*

loss䇆?�{<-       �	M�1*u��A�*

loss3��?G���       �	g*H/u��A�*

loss��?+�k       �	��]4u��A�*

loss�j?qc0�       �	�p9u��A� *

loss���?��%�       �	Kͱ@u��A� *

loss���?	�-�       �	��Eu��A� *

loss~f�?+RH       �	<j�Ju��A� *

loss�ֻ?2       �	���Ou��A� *

lossQx�?%�       �	� Uu��A� *

loss�B�?����       �	�Zu��A� *

loss1F�?�Lp�       �	��!_u��A� *

loss|@��e�       �	�%+du��A� *

loss��?�,0�       �	�'Eiu��A� *

lossC3�?���X       �	��\nu��A� *

loss�΍?Ȃ�.       �	��uu��A� *

loss�`�?���<       �	8��zu��A� *

loss:Uh?���G       �	�$�u��A�!*

lossdun?zӷ       �	�Ӹ�u��A�!*

lossͩ?_H       �	�2��u��A�!*

loss�=�?���       �	��̎u��A�!*

loss�V�?�k��       �	�eޓu��A�!*

loss�<�?	�v       �	4��u��A�!*

loss+�h?�rv�       �	pDÞu��A�!*

loss�G�?I�#       �	o�u��A�!*

lossZ�@_�       �	�F?�u��A�!*

lossɍ�?�� �       �	�R��u��A�!*

lossT3O?<o-�       �	){��u��A�!*

loss�Z�?eJG�       �	t#��u��A�!*

lossĚ,@�A�       �	7R��u��A�!*

loss74�?�%�       �	���u��A�"*

loss �?s?�       �	<���u��A�"*

loss"=`?&z�g       �	8���u��A�"*

loss�h�?�~cZ       �	�u��u��A�"*

loss�)�?�B       �	�J��u��A�"*

lossL�?���1       �	ڪ"�u��A�"*

loss�o�?�� 4       �	��'�u��A�"*

lossM��?�B�       �	̸5�u��A�"*

loss��?���       �	BJ�u��A�"*

loss�5�?�0�       �	d<R�u��A�"*

lossKȟ?/H��       �	��c�u��A�"*

loss �@��n       �	3Qr�u��A�"*

loss�ݑ??P=       �	��zv��A�"*

loss?p�?q��       �	���	v��A�#*

loss�O?��n       �	"Ëv��A�#*

lossjv?KCJ�       �	�+�v��A�#*

loss��?Q� �       �	v	v��A�#*

lossf��?�^�       �	y� v��A�#*

loss���?0�ݥ       �	�%v��A�#*

loss6͚?�)e�       �	DP-*v��A�#*

loss3Z�?f�P       �	8gS/v��A�#*

loss��?�9�       �	9�i4v��A�#*

loss)/�?#��       �	�a�9v��A�#*

loss��?���       �	�[�>v��A�#*

loss���?�w�s       �	;ĘCv��A�#*

loss���?F:�+       �	L��Jv��A�$*

lossְ�?0�)�       �	�9�Ov��A�$*

loss�-r?��(J       �	r��Tv��A�$*

loss���?H�d#       �	T�Yv��A�$*

loss���??�       �	�_v��A�$*

loss2| @���       �	}�!dv��A�$*

loss�{?���       �	�J1iv��A�$*

lossj1{?v���       �	��Fnv��A�$*

loss=�?�t"�       �	� dsv��A�$*

loss�X�?.�b�       �	8�gxv��A�$*

loss��?wV#�       �	m��v��A�$*

loss���?΃�j       �	�[��v��A�$*

loss�E�?~�v       �	E,�v��A�$*

loss�@�?�$5G       �	5B�v��A�%*

loss�D�?AAx�       �	F@=�v��A�%*

lossJ��?��       �	�W�v��A�%*

loss���?��       �	l!e�v��A�%*

lossqe�?�[��       �	��p�v��A�%*

loss�$b?��4       �	�=y�v��A�%*

loss�{�?�i*+       �	\T��v��A�%*

loss���?:�(+       �	_��v��A�%*

lossUL$@���       �	�&��v��A�%*

loss���?����       �	�Ⱦv��A�%*

loss�n	@̔Q�       �	p���v��A�%*

loss���?�w�       �	���v��A�%*

loss��3@���       �	�4��v��A�%*

loss?�?�M�m       �	�Z�v��A�&*

loss.�@���       �	;�
�v��A�&*

loss�?χt       �	��v��A�&*

loss���?~L       �	p	*�v��A�&*

loss�
@ʌ�       �	˺R�v��A�&*

loss���?kA�x       �	�^�v��A�&*

loss��?.\��       �	U�o�v��A�&*

lossί�?�y��       �	�v}�v��A�&*

loss�V�?�}m�       �	yW��v��A�&*

loss2
r?���       �	1�w��A�&*

loss�ݮ?@�m       �	�w��A�&*

lossà�?&�       �	���w��A�&*

loss.Q�?��       �	�%�w��A�&*

losshߌ?�T�=       �	�^�w��A�'*

lossż?�2E       �	�Bw��A�'*

loss�ã?P�U�       �	
�#w��A�'*

loss�ng?[~Jo       �	��)(w��A�'*

loss��?��Cf       �	vnE-w��A�'*

loss	�?'�{       �	�CK2w��A�'*

loss���?�4�       �	 }[7w��A�'*

loss>�?x'�z       �	{o<w��A�'*

loss���?U�       �	��~Aw��A�'*

loss���?ɯ�`       �	&R�Fw��A�'*

loss�t�?W��9       �	��Kw��A�'*

loss�X�?�@�w       �	�m�Rw��A�'*

loss��?Vw       �	T��Ww��A�'*

loss#?�?v�S�       �	!=�\w��A�(*

lossM*�?��       �	���aw��A�(*

loss�x�?oӌ�       �	,dgw��A�(*

loss�)o?� K�       �	iolw��A�(*

loss��?�;�       �	�|-qw��A�(*

loss}��?\�z�       �	EcCvw��A�(*

loss�@G X       �	�;[{w��A�(*

loss�?�u/r       �	�of�w��A�(*

losspf?�e��       �	�i��w��A�(*

loss���?�0v�       �	�E��w��A�(*

loss��?U~D       �	E��w��A�(*

loss� �?e�,       �	 u̖w��A�(*

loss?�?L���       �	�vћw��A�)*

loss?�?���[       �	���w��A�)*

loss�?h�m�       �	_$�w��A�)*

loss�=m?�/��       �	���w��A�)*

lossZ�7?SJ1       �	���w��A�)*

loss��|?�ć�       �	V�w��A�)*

loss<de??^q�       �	���w��A�)*

loss��?��\s       �	�6��w��A�)*

loss zl?l�       �	ٔ��w��A�)*

loss��?�6�       �	^��w��A�)*

loss�L�?�� U       �	���w��A�)*

loss�D??M��       �	ع��w��A�)*

loss(y�?�F�t       �	���w��A�)*

loss���?�])+       �	�|��w��A�**

loss���?�QI�       �	~���w��A�**

loss�j�?=븓       �	�]��w��A�**

loss"1W?�et�       �	�LI�w��A�**

loss͸�?��       �	�tW�w��A�**

lossH�a?{���       �	��a�w��A�**

loss�8�?6�       �	��q x��A�**

loss�ݡ?�x!�       �	j��x��A�**

lossk�?d�\       �	I��
x��A�**

lossr��?X�%�       �	�%�x��A�**

lossH�]?1�t�       �	�x��A�**

lossa�?����       �	9+�x��A�**

loss�p?��1       �	��x��A�**

loss��?��!�       �	&�$&x��A�+*

lossά�?�n�}       �	c)E+x��A�+*

lossF�?<�       �	�e0x��A�+*

lossw�0@�K�_       �	K��5x��A�+*

losst��?��~�       �	F��:x��A�+*

loss�Қ?�Gn       �	��?x��A�+*

lossвt?Ah�4       �	�Dx��A�+*

loss�Ө?T#�D       �	�j�Ix��A�+*

loss؇�?�q       �	\�Nx��A�+*

loss�M�?��O�       �	ۆ�Sx��A�+*

lossp�u?C�kA       �	��
[x��A�+*

loss�r�?m�G       �	x)`x��A�+*

loss36�?p��       �	��Oex��A�+*

loss,�h?l�*r       �	�rajx��A�,*

lossb�?O�T�       �	�f�ox��A�,*

loss�vl?���J       �	*ʒtx��A�,*

lossT΂?�T+�       �	9�yx��A�,*

loss��N?��Uw       �	9%�~x��A�,*

loss��? �7y       �	����x��A�,*

lossTdp?xTP       �	�\��x��A�,*

loss9��?$�=       �	+N�x��A�,*

loss��?��#       �	�^�x��A�,*

loss\�?r�       �	a�x��A�,*

lossJ��?����       �	.#�x��A�,*

loss(�@,�L        �	ĳ0�x��A�,*

loss�C�?��#       �	<L2�x��A�,*

lossX�y?���       �	@�x��A�-*

lossǒ�?�_       �	�IK�x��A�-*

loss��?T,N       �	&�[�x��A�-*

loss�i?!�b�       �	'�i�x��A�-*

loss��?���       �	���x��A�-*

loss�Q�?t��S       �	�3��x��A�-*

loss���?���       �	ׅ��x��A�-*

lossx\?���       �	c���x��A�-*

loss��m?S�P�       �	_'�x��A�-*

loss���?W�s       �	���x��A�-*

loss��?ӗ<       �	��1�x��A�-*

loss�g�?Ш)�       �	��8�x��A�-*

loss\"r?g@I�       �	�C�x��A�.*

loss�jM?kt       �	Ova�x��A�.*

loss:��?��Up       �	I�y�x��A�.*

loss�V�?Z8�X       �	6u��x��A�.*

loss�g<?���       �	{��y��A�.*

loss���?�,�	       �	(E�y��A�.*

lossٜ�?�ْ       �	���y��A�.*

loss���?ϗw9       �	�O�y��A�.*

loss��?�l,       �	]��y��A�.*

loss�_�?$蝟       �	���y��A�.*

loss��e?o$t�       �	K�!y��A�.*

loss\�H?�R�c       �	B�&y��A�.*

loss���??��       �	��*.y��A�.*

loss��@�{�       �	�T93y��A�/*

loss��q?WB�       �	��N8y��A�/*

lossiM�?�r1�       �	n��=y��A�/*

loss��?���y       �	H�Cy��A�/*

loss��@y       �	�#;Iy��A�/*

loss=��?�H�       �	/��Ny��A�/*

loss���?t���       �	���Sy��A�/*

loss��?�@�t       �	��Yy��A�/*

loss���?�<@�       �	c�^y��A�/*

loss��?�X�       �	u�7ey��A�/*

loss�w�?����       �	�djy��A�/*

loss�7�?uݩd       �	��zoy��A�/*

loss8ٰ?�A�       �	��ty��A�/*

lossS�?'.m�       �	ni�yy��A�0*

loss��?��=       �	��~y��A�0*

loss>�E?����       �	����y��A�0*

loss���?��*       �	�_ֈy��A�0*

loss�?�?����       �	^Iݍy��A�0*

lossa�?��*�       �	\�y��A�0*

loss��?���       �	&S�y��A�0*

loss?��?�@.b       �	�#�y��A�0*

loss^G�?��:       �	ΨB�y��A�0*

loss��@u���       �	]�T�y��A�0*

loss��?U�       �	!#`�y��A�0*

loss��?z��       �	�mn�y��A�0*

lossF�J?�<��       �	�4v�y��A�0*

loss��d?�hMd       �	���y��A�1*

lossIl?~�C       �	��y��A�1*

loss-ܓ?:�4       �	A��y��A�1*

lossn�q?CI��       �	Kt
�y��A�1*

loss.�?M�C	       �	�#�y��A�1*

loss�{?�3.       �	��-�y��A�1*

lossh�y?Tg�       �	�+;�y��A�1*

loss�Rp?���
       �	f�M�y��A�1*

loss�u?[�;       �	<�k�y��A�1*

lossr�~?	�ܩ       �	׿x�y��A�1*

loss3�?.<'       �	9���y��A�1*

lossu�?RG�Y       �	_%��y��A�1*

loss���?��SX       �	�Ɛ�y��A�1*

loss.9c?��ʃ       �	���z��A�2*

loss0�?�<$u       �	)>�z��A�2*

loss�נ?@\�       �	�,�z��A�2*

loss��}?�BW�       �	ö�z��A�2*

loss�Չ?}��       �	R,�z��A�2*

loss�9�?�~P       �	 �z��A�2*

loss��U?<��i       �	��"z��A�2*

loss�S�?�E��       �	,J''z��A�2*

loss��?b�"k       �	��>,z��A�2*

loss���?�\�1       �	�2L1z��A�2*

loss���?钐y       �	Y�!9z��A�2*

loss�j?��7V       �	o��=z��A�2*

losscj?�ڄ�       �	R)�Bz��A�3*

loss��j?�۫�       �	��4Gz��A�3*

loss>f?º�7       �	�5�Kz��A�3*

loss~��?��Y%       �	��ZPz��A�3*

loss�/?�~�       �	]��Tz��A�3*

loss�Ѡ?l�0u       �	G�sYz��A�3*

loss�	�?��.f       �	A�]z��A�3*

loss��^?+̚|       �	�k�bz��A�3*

loss:|�?�a��       �	���hz��A�3*

lossK�Z?
�t       �	�0mz��A�3*

loss?V�?� �       �	�Vrz��A�3*

loss��?���       �	jN�vz��A�3*

loss�N?�]�H       �	~� {z��A�3*

loss^�(?���       �	��z��A�4*

loss$ɵ?��FI       �	� �z��A�4*

lossD��?��       �	�Ȇ�z��A�4*

lossn�?#��\       �	���z��A�4*

lossLw�?�:R�       �	����z��A�4*

loss��)?[��v       �	����z��A�4*

loss>�?��<�       �	P ��z��A�4*

loss�љ?XOh       �	��z��A�4*

loss�_?C?       �	�I��z��A�4*

loss_a?q��       �	Y2�z��A�4*

lossLbg?�E<I       �	� ��z��A�4*

loss��5?�B_�       �	�t/�z��A�4*

loss$h?���J       �	<k��z��A�4*

loss׈�?IIk8       �	��D�z��A�5*

loss6�s?�s��       �	����z��A�5*

lossg:�?1!N�       �	N�?�z��A�5*

loss�e>?�~<{       �	By��z��A�5*

loss��?@!�6       �	hn�z��A�5*

loss�&r?���       �	| �z��A�5*

loss���?�%]�       �	���z��A�5*

lossEG?��k;       �	5�z��A�5*

loss>K�?/�V       �	����z��A�5*

loss,�X?�CTC       �	�29�z��A�5*

loss�sr?�       �	����z��A�5*

lossZ��?gb`�       �	��I�z��A�5*

loss��t?�OW       �	����z��A�5*

loss29:?����       �	�N�z��A�6*

loss��?+z�       �	����z��A�6*

loss �k?/z·       �	�9f{��A�6*

lossZ-Y? ��       �	���{��A�6*

loss�ed?X���       �	�jr{��A�6*

loss�l�?ڥYD       �	N��{��A�6*

loss�Ǫ?T�.�       �	vށ{��A�6*

loss�U�?���       �	v�{��A�6*

lossZ�?]       �	���{��A�6*

lossE��?�?�       �	ܺ$&{��A�6*

loss���?vd{�       �	�+�*{��A�6*

loss���?q�       �	��5/{��A�6*

loss���?Bgp|       �	�!�3{��A�6*

lossL�,?�A��       �	�>S8{��A�7*

loss#!�?��eH       �	j��<{��A�7*

loss��Q?̀`�       �	�$qA{��A�7*

loss��s?�UV�       �	�L F{��A�7*

loss���?����       �	{תJ{��A�7*

loss��?&���       �	|GCO{��A�7*

loss�p?�.-       �	T��U{��A�7*

loss<t�?�.</       �	�?Z{��A�7*

loss���?��       �	�)�^{��A�7*

loss�Ō?�$��       �	�Uc{��A�7*

loss5a�?aԶc       �	en�g{��A�7*

loss�K?�V�>       �	>ywl{��A�7*

loss4B�?��@\       �	�@q{��A�8*

lossu4�?�Eh�       �	쥤u{��A�8*

loss�&q?^�       �	��'z{��A�8*

loss_��?��       �	 �~{��A�8*

loss��?��       �	O%�{��A�8*

loss���?z�O�       �	�g��{��A�8*

loss�,�?��n       �	�7�{��A�8*

loss�@�?c"[       �	�͒{��A�8*

loss,"?V>IE       �	b�Z�{��A�8*

lossv_�?3�R�       �	��{��A�8*

lossVT�?�,�       �	J�g�{��A�8*

loss�A�?��u       �	��{��A�8*

loss|�?�a�       �	�a��{��A�8*

loss�َ?�n�       �	qr�{��A�9*

lossT�"?��       �	��޴{��A�9*

loss�A�?�4��       �	|_n�{��A�9*

lossVp?+&�       �	�'��{��A�9*

loss?S��       �	���{��A�9*

loss(r�?ԞD       �	k�"�{��A�9*

loss��e?��b       �	٬�{��A�9*

lossַ}?�&>�       �	��6�{��A�9*

loss]6�?=\t       �	�k��{��A�9*

loss��?�_P       �	�W�{��A�9*

loss�,Q?3�U'       �	���{��A�9*

loss`>�?ZTm%       �	2�x�{��A�9*

loss�>�?T��)       �	��	�{��A�9*

loss��b?����       �	9��{��A�:*

loss�i\?E�w~       �	���{��A�:*

loss�g�?ȗ�       �	�"��{��A�:*

loss��{?��"�       �	c|A�{��A�:*

loss0�H?/D�       �	�&��{��A�:*

lossL�g?+�J�       �	��P|��A�:*

loss
�y?Hh�       �	���|��A�:*

loss.܋?�eSi       �	�{d|��A�:*

loss޻?V���       �	|��A�:*

loss��n?44�        �	�q�|��A�:*

loss�`�?����       �	<|��A�:*

loss���?bS+(       �	��!|��A�:*

loss��?       �	P�'&|��A�:*

loss B�?ـ�X       �	q�*|��A�;*

loss��?���       �	ʈE/|��A�;*

loss�)Q?�9�       �	e��3|��A�;*

loss�?�s!       �	��Y8|��A�;*

losst�D?�p8�       �	���<|��A�;*

loss�>/?���       �	�gC|��A�;*

loss3��?�r�^       �	���G|��A�;*

loss>~?���$       �	�d�L|��A�;*

loss�M?�cC�       �	?r$Q|��A�;*

lossD�^?0,P"       �	��U|��A�;*

loss�;?;:�       �	��9Z|��A�;*

lossP�u?�à       �	Tn�^|��A�;*

loss*�?��       �	�`c|��A�;*

loss�d|?��.�       �	�)�g|��A�<*

loss�Vg?%���       �	�?�l|��A�<*

loss��?�bV~       �	S$ s|��A�<*

loss�O�?Gͬ�       �	Cs�w|��A�<*

loss��?��xO       �	��L||��A�<*

loss�ʖ?Ht��       �	$�ـ|��A�<*

loss��e?�N�T       �	t�e�|��A�<*

loss��E?�h       �	��|��A�<*

loss	H�?ni@       �	i�v�|��A�<*

loss�ٜ?ěCP       �	Tq�|��A�<*

loss�~�?oހF       �	����|��A�<*

lossJ��?#H6�       �	1
�|��A�<*

loss��?�B\�       �	{���|��A�=*

loss��]?/��p       �	9 �|��A�=*

loss$��?+�       �	7���|��A�=*

lossh)�?v�<y       �	`wA�|��A�=*

lossD�@Z׾�       �	>�ִ|��A�=*

loss���?Z�TV       �	� ]�|��A�=*

loss��x?M�W       �	m��|��A�=*

loss��@ܾc       �	rms�|��A�=*

loss�k�?�o�       �	��|��A�=*

lossh��?Om�       �	x��|��A�=*

loss(��?X+**       �	�t�|��A�=*

loss��?�       �	nާ�|��A�=*

loss��?��[�       �	�9t�|��A�=*

loss�6�?:��       �	I��|��A�>*

loss$��?(��       �	+���|��A�>*

loss�Y1?ū/�       �	dy��|��A�>*

loss]A�?/��       �	_$��|��A�>*

loss�q�?㵶�       �	Nb:�|��A�>*

loss+j?��q       �	�	��|��A�>*

loss6t?R^�       �	�pM�|��A�>*

loss0��?�R~u       �	�}��A�>*

loss��?�R�       �	��\}��A�>*

loss��R?���       �	���}��A�>*

loss��?7D�       �	�u}��A�>*

loss��?#DC�       �	7��}��A�>*

loss�>�?���       �	��|}��A�>*

loss`^k?.�5:       �	v� }��A�?*

loss2�i?�A3�       �	���#}��A�?*

lossȱP?��%       �	X<(}��A�?*

lossZ#?M�)       �	c{�,}��A�?*

loss��?ɵ`       �	�\3}��A�?*

lossغ�?57       �	�r�7}��A�?*

loss:�p?�GtM       �	2$<}��A�?*

lossv-@�Z|�       �	�ָ@}��A�?*

lossV�?�v�R       �	�@E}��A�?*

loss�B=?�N?       �	�y�I}��A�?*

lossOa�?�Q��       �	]PhN}��A�?*

loss�oA?���       �	Ի�R}��A�?*

loss0��?VG+�       �	�tW}��A�?*

loss���?���       �	��\}��A�@*

loss��?y���       �	,�b}��A�@*

loss j?��_       �	h@g}��A�@*

lossC̀?&6�       �	2�k}��A�@*

loss�JH?A_�s       �	��,p}��A�@*

loss�W?|AL�       �	hv�t}��A�@*

lossz�??'F�       �	)�Wy}��A�@*

loss���?D���       �	�p�}}��A�@*

loss��?"S
�       �	��u�}��A�@*

loss�� ?Gk��       �	˽��}��A�@*

lossե�?7��       �	8H��}��A�@*

lossj��?�Y"%       �	<�'�}��A�@*

loss'��?g/�]       �	���}��A�@*

loss�#�>�
��       �	�8�}��A�A*

loss��?��P<       �	�	��}��A�A*

loss,g?R(w|       �	�Q�}��A�A*

loss�V`?!��       �	��ݨ}��A�A*

loss�HQ?=��B       �	��c�}��A�A*

loss�n�?o+�       �	z�}��A�A*

loss�??��9       �	�6�}��A�A*

lossV7?g�0Q       �	&T�}��A�A*

loss�:t?��$       �	�H��}��A�A*

loss~�3?��       �	f��}��A�A*

loss&n?�f��       �	���}��A�A*

loss�:�?����       �	|G�}��A�A*

loss�q�?1I]�       �	:��}��A�B*

loss^�8?���       �	�_�}��A�B*

loss��7?�M�o       �	�j�}��A�B*

loss��?���&       �	�ZG�}��A�B*

lossv��?��U       �	U3D�}��A�B*

loss���?/*$�       �	�G�}��A�B*

lossJ?�BTa       �	��i�}��A�B*

loss�5?�d�       �	�5s�}��A�B*

loss^1?�<�g       �	�!�}��A�B*

lossV�/?5��       �	�~��A�B*

loss~��?g�f       �	��	~��A�B*

loss��q?ē       �	}�~��A�B*

loss�v�?��=T       �	nm�~��A�B*

loss2u0?����       �	IM�~��A�C*

lossc��?�1       �	3T�~��A�C*

loss�LL?߫�       �	��"~��A�C*

loss��T?��(�       �	=*~��A�C*

loss�]?�/�       �	�{/~��A�C*

loss(�?&�*�       �	J.4~��A�C*

loss �s?rR��       �	-�29~��A�C*

loss�b�?���G       �	�tE>~��A�C*

loss?�`�       �	d�TC~��A�C*

loss�_�?3�E       �	�y_H~��A�C*

lossSg?��       �	�MvM~��A�C*

loss�0?q�       �	V)�R~��A�C*

lossl�%?
�q       �	���W~��A�C*

loss�??�t�U       �	�'�^~��A�D*

loss��L?E�p       �	Jy�c~��A�D*

loss��;??�       �	W��h~��A�D*

lossb��?�\?�       �	�m~��A�D*

lossx?熵9       �	Y��r~��A�D*

lossy��?���       �	7x~��A�D*

lossc�?���Q       �	H>}~��A�D*

loss�(�?���       �	x�C�~��A�D*

loss��?0��N       �	`�U�~��A�D*

losslX�?k %^       �	GUd�~��A�D*

loss�P?N��k       �	����~��A�D*

loss��;?$y�       �	,}��~��A�D*

loss�ό?`i^       �	�~��A�D*

loss�j�? �        �	찢~��A�E*

loss���?�*�       �	��ħ~��A�E*

loss�;�?�.D}       �	S	ˬ~��A�E*

loss���?�ˤ       �	�Qб~��A�E*

loss� ]?���       �	�Lն~��A�E*

loss�?=��_       �	8��~��A�E*

loss~�x?�ddM       �	8���~��A�E*

lossp��?^¯       �	���~��A�E*

loss8�/?4��       �	/��~��A�E*

loss��?��&�       �	�2�~��A�E*

loss�[?e��s       �	:�B�~��A�E*

lossU��?!M��       �	X�Q�~��A�E*

loss-p�?��o�       �	r�X�~��A�E*

loss�Re?n�       �	� `�~��A�F*

lossY�?3��>       �	��u�~��A�F*

loss�7g?�ˢ�       �	.S��~��A�F*

lossR09?���       �	*R��~��A�F*

loss�NN?�e�$       �	�f��~��A�F*

lossX�[?���J       �	Q����A�F*

lossr�|?�g*>       �	+L���A�F*

lossOh�?�d�n       �	@����A�F*

loss�~j?�lof       �	\9���A�F*

lossK�?'oB       �	�n+��A�F*

loss^��?���       �	a5Q��A�F*

lossq��?j���       �	��e ��A�F*

lossF��?���+       �	,r%��A�G*

loss��?��ܘ       �	���*��A�G*

loss�l?]=��       �	�v	2��A�G*

loss�'�?A�A       �	}=+7��A�G*

loss��b?���       �	E<��A�G*

lossY��?Ȯ=�       �	��TA��A�G*

loss��g?��       �	haF��A�G*

loss�%�?��{�       �	`szK��A�G*

loss���?1�`�       �	��P��A�G*

loss���?�Wmw       �	6?�U��A�G*

lossm��?�[�u       �	��Z��A�G*

loss("?�!/�       �	��_��A�G*

loss�C�?R�Խ       �	�g��A�G*

lossfˁ?�BMg       �	ʩ1l��A�H*

lossȜG?1��V       �	KVBq��A�H*

loss�И?��Ry       �	{_v��A�H*

loss�,?ц"       �	qX�{��A�H*

loss��S?�T�       �	m<����A�H*

loss��~?�5�       �	�����A�H*

loss`�V?��       �	�˩���A�H*

loss��}?%�       �	V�����A�H*

loss�'L?��y       �	�HΔ��A�H*

lossN5�?��7       �	�f����A�H*

loss#�?���       �	D����A�H*

lossf��?��K�       �	����A�H*

loss�Β?Zȱ�       �	1�:���A�H*

loss�0|?:j{�       �	��P���A�I*

loss��e?��$E       �	j�u���A�I*

lossi�?�p�       �	������A�I*

loss ��?�N�       �	�J����A�I*

loss�~�?�CH*       �	[����A�I*

loss�+\?�0       �	s+����A�I*

loss{.�?��       �	w.����A�I*

loss�&�?���       �	b����A�I*

loss���?]`�       �	hv;���A�I*

loss�g?��Sx       �	��G���A�I*

loss jo?�+һ       �	��e���A�I*

loss�m0?��XF       �	O����A�I*

loss<ߐ?%J'`       �	�ڔ���A�I*

loss��f?�	f�       �	������A�J*

loss�#�?V�mN       �	������A�J*

loss���?@�+�       �	e�����A�J*

loss�S�?ʮ�       �	Z����A�J*

lossΏ�?	�wL       �	h�+���A�J*

lossF(v?���       �	.�>���A�J*

loss�k�?l#�       �	�N���A�J*

loss,4?`\9>       �	��f���A�J*

loss�2P?�kH�       �	������A�J*

losss��?�(��       �	I��$���A�J*

loss��?���       �	|��)���A�J*

loss���?T.��       �	�,�.���A�J*

loss�M?��U�       �	�;�3���A�J*

lossO�?T��       �	��8;���A�K*

loss`�b?�Lɰ       �	��Q@���A�K*

loss{_?G�5�       �	WAgE���A�K*

lossH�}?�l��       �	
i�J���A�K*

loss&��?��J       �	G �O���A�K*

loss��P?����       �	\��T���A�K*

loss,��?5�<       �	�Y���A�K*

lossR�7?BH��       �	��^���A�K*

loss�G�?[,�~       �	?�d���A�K*

lossޘ>?�>=2       �	;�i���A�K*

loss���?��jJ       �	Ip���A�K*

loss�^?���       �	�_ju���A�K*

loss"�n?�#@       �	\��z���A�L*

loss��Q?Pk6       �	�-}����A�L*

loss���?��7�       �	�J����A�L*

lossޑ/?���       �	(�܋���A�L*

loss*�@?�+5C       �	V�E����A�L*

lossD'G?T�"S       �	$ш����A�L*

lossH-}?�L6�       �	�Ǵ����A�L*

lossȊ\?�T�G       �	"�Ƞ���A�L*

loss�>S?��C�       �	:�燎��A�L*

loss� �?1sG       �	������A�L*

loss�^?؆Z       �	�2����A�L*

loss
�?��5{       �	��?����A�L*

loss0�,?��H$       �	��W����A�L*

lossV8P?ե�       �	��p����A�M*

loss��T?�-       �	���ƀ��A�M*

loss��C?�?h�       �	�~�ˀ��A�M*

loss<�5?��,       �	'��Ѐ��A�M*

loss��?f��       �	3��Հ��A�M*

loss��/?��s       �	�I݀��A�M*

loss�6?	�Z       �	��#‘�A�M*

lossaH�?��       �	U9瀘�A�M*

loss�7L?AW;       �	�hU쀘�A�M*

losspK�>���       �	�{��A�M*

loss|�8?n��3       �	"r�����A�M*

lossf�g?�!q>       �	z�����A�M*

loss�
0?+�f3       �	`� ���A�M*

loss�m?H;       �	�g����A�N*

lossx�?�1�J       �	W%�
���A�N*

loss0"0?d��|       �	�4���A�N*

lossn"�?��       �	zE���A�N*

loss<D-?�g�       �	�@\���A�N*

loss�+U?���       �	 �h!���A�N*

loss��V?s       �	ZI~&���A�N*

loss�w	?��D       �	��+���A�N*

loss��?X�^	       �	��0���A�N*

loss�Bu?����       �	^��5���A�N*

loss���?PH�_       �	V��:���A�N*

loss�v?�S�       �	;��?���A�N*

lossƯj?
�\       �	��#G���A�N*

lossbka?�k��       �	�9FL���A�O*

loss�2@�4�h       �	9~bQ���A�O*

loss���?Xm2       �	�ێV���A�O*

lossa7�?/�       �	��[���A�O*

loss��W?�K0       �	�i�`���A�O*

loss��]?pʜ�       �	�x�e���A�O*

loss�?��H       �	~��j���A�O*

loss>�)?v��       �	�"p���A�O*

loss4B?T�       �	��)u���A�O*

lossnx?���n       �	�HZ|���A�O*

lossfc?ّ��       �	��z����A�O*

loss:ZH?&��Y       �	�M�����A�O*

lossp:�?7��       �	�������A�O*

loss��O?�{��       �	EHِ���A�P*

lossf+A? ��       �	gG�����A�P*

lossN�8?[��       �	������A�P*

lossT	}?E���       �	�����A�P*

losszQ?�(2       �	8.5����A�P*

lossX�\?'3H�       �	JCJ����A�P*

loss@�;?O�T       �	R*z����A�P*

loss�?���"       �	1{�����A�P*

loss9�?]��       �	r�����A�P*

lossBkf?Lj��       �	������A�P*

loss�Y?���#       �	6"�Ł��A�P*

lossx�G?���       �	1`�ʁ��A�P*

loss��r?�LH1       �	��ρ��A�Q*

loss�Eq?E���       �	alՁ��A�Q*

loss��?�       �	C9 ځ��A�Q*

loss�^?)<f       �	�He߁��A�Q*

loss0�]?����       �	��恘�A�Q*

loss4�T?�h       �	tԫ끘�A�Q*

loss���?��       �	E������A�Q*

lossG�?Ǔ��       �	`������A�Q*

loss9W�?���       �	}%����A�Q*

lossXف?<Q��       �	��0 ���A�Q*

loss��?���9       �	��I���A�Q*

loss��r?��O�       �	��Y
���A�Q*

loss��?�M~�       �	�u���A�Q*

loss�߃?���       �	d����A�R*

loss*�f?�/k       �	�����A�R*

lossXfC?x�?b       �	�� ���A�R*

loss�au?%2H�       �	�q�%���A�R*

lossެ9?$1�       �	�X�*���A�R*

losst�<?Z�       �	0���A�R*

loss��c?�@       �	}�5���A�R*

loss��?Q��S       �	�L-:���A�R*

loss�?�T\�       �	xcH?���A�R*

loss��i?-m,�       �	�ZD���A�R*

lossF�??M��       �	�~I���A�R*

lossLh�?h91I       �	�I�P���A�R*

loss��g?FE�l       �	Y��U���A�R*

lossh�D?z7��       �	��Z���A�S*

loss�ڑ?�-t�       �	'��_���A�S*

loss�1,?�D^       �	�e���A�S*

loss���?��+�       �	��j���A�S*

loss�|?��#       �	Н'o���A�S*

loss��.?,�y�       �	F�Gt���A�S*

loss�e?===�       �	C�Wy���A�S*

loss�d?��       �	 �r~���A�S*

loss�Kx?6�<U       �	������A�S*

losst#S?�z�       �	��Ɋ���A�S*

loss�?#��M       �	�H⏂��A�S*

loss|o�?��       �	u�򔂘�A�S*

loss H�?�H�       �		����A�S*

loss�?4�	�       �	+�)����A�T*

loss2?�?�LT1       �	��N����A�T*

lossP`?��W�       �	e�`����A�T*

loss��l?���       �	��r����A�T*

losso�?qcR�       �	qY�����A�T*

loss��k?�Zi�       �	q�׺���A�T*

loss��?�d�s       �	)	����A�T*

loss��?b9�7       �	�Jł��A�T*

loss��"@�\��       �	�77ʂ��A�T*

loss�C"?C1       �	KwNς��A�T*

loss�\3?���       �	M�`Ԃ��A�T*

loss�V�?���       �	�ق��A�T*

loss�)U?��        �	��ނ��A�T*

loss|&O?>a�       �	$ս゘�A�U*

loss�2?g.?       �	��肘�A�U*

loss�P?��(       �	 ����A�U*

loss�8t?\�p       �	Ψ����A�U*

loss��>?h"��       �	Z�'����A�U*

loss�$?�DC       �	�w9����A�U*

loss(�?�BT~       �	shZ���A�U*

loss��S?����       �	<3	���A�U*

loss#��?P       �	�F����A�U*

loss^YJ?8Li]       �	�����A�U*

loss<vw?v0�       �	oյ���A�U*

lossf�*?8���       �	ʉ����A�U*

lossZ@�?U��E       �	�e�$���A�V*

lossw��?u��p       �	���)���A�V*

loss���?�oe       �	��/���A�V*

loss�4?%'3�       �	�6>4���A�V*

lossՕ?��Ǫ       �	�.X9���A�V*

loss^du?:T2       �	^�m>���A�V*

loss��l?��ZI       �	=a~C���A�V*

loss��r?���       �	�ֈH���A�V*

lossh��?Ӷ!)       �	\8�M���A�V*

loss�g
@�3p       �	1�R���A�V*

loss���?k��       �	�O�Y���A�V*

lossN�P?N-Y       �	F��^���A�V*

losszc<?V�31       �	X�c���A�V*

loss\^H?��@�       �	Z�i���A�W*

loss�7?���t       �	�}$n���A�W*

loss9�?�4�       �	3s���A�W*

loss?q N�       �	I�=x���A�W*

loss�g@�.�B       �	�C}���A�W*

loss�K�?)�e�       �	��N����A�W*

lossdjo?��#3       �	{p����A�W*

loss}?�?>'��       �	/������A�W*

loss<g`?�P��       �	�������A�W*

loss��?=0G�       �	QM�����A�W*

loss1=�?�\~�       �	d�͝���A�W*

loss�I�?'���       �	z�ࢃ��A�W*

lossZX�?�[�       �	2򧃘�A�W*

loss�W�?t<��       �	�'����A�X*

loss&�?�f@?       �	�0<����A�X*

loss�N?q}�S       �	�O����A�X*

losst�k?�^�       �	�NZ����A�X*

loss��E?q�4�       �	�ɢÃ��A�X*

loss�3�?f�d       �	S�ȃ��A�X*

losss�i?�չB       �	���̓��A�X*

loss��?�6       �	�a�҃��A�X*

lossb�v?�ǳw       �	���׃��A�X*

loss�P�?����       �	 �݃��A�X*

loss��m?-�       �	�⃘�A�X*

loss;��?���       �	��烘�A�X*

lossH�Z?�I�n       �	�HF샘�A�X*

loss�J?�4X       �	��V��A�Y*

lossn�[?���d       �	�?p����A�Y*

lossZ�%?�_��       �	�b�����A�Y*

loss4�\?���r       �	�����A�Y*

loss�[?�w�-       �	������A�Y*

loss&G?�_�       �	0�����A�Y*

lossܻh?�~J       �	�����A�Y*

loss�ԅ?S��       �	�����A�Y*

lossN�f?+�FU       �	��7���A�Y*

loss��?m+�h       �	��!���A�Y*

loss$��?��c+       �	.�~'���A�Y*

lossb�V?����       �	��?/���A�Y*

loss6�K?)��f       �	�4���A�Y*

loss(?$H�x       �	o��9���A�Z*

loss*jx?�E��       �	ڭ�>���A�Z*

lossf�U?�\��       �	�S�C���A�Z*

loss2�z?��       �	"�I���A�Z*

loss�&??���       �	�#N���A�Z*

lossRwK?՚�       �	��/S���A�Z*

loss ><?qH�m       �	vRJX���A�Z*

loss>�Y?�? �       �	� \]���A�Z*

loss@�w?k�       �	r��d���A�Z*

lossĝ?����       �	�ޡi���A�Z*

loss��n?�D�-       �	���n���A�Z*

lossFߌ?ĢԶ       �	�/�s���A�Z*

lossI~�?!��x       �	��x���A�[*

loss��A?��m�       �	��}���A�[*

loss4�"?<���       �	O�����A�[*

loss��2?uΟ�       �	ƾ
����A�[*

loss	ʶ?�4E>       �	|F����A�[*

loss��b?�l-       �	��/����A�[*

lossJ2�?u�u       �	�V�����A�[*

loss��?#��       �	�­����A�[*

loss�*?!��[       �	��ģ���A�[*

lossl(x?���       �	eި���A�[*

loss��7?C�l�       �	r�����A�[*

loss5��?���       �	��%����A�[*

lossB�I?u��       �	l<����A�[*

loss.Y6?3ق�       �	�O����A�\*

loss�[?,O�*       �	��a��A�\*

loss��E?".�
       �	}�|Ǆ��A�\*

loss�?�T�       �	��΄��A�\*

loss�3~?}��       �	���ӄ��A�\*

lossr6?O:       �	>�{؄��A�\*

loss�?ȸV       �	�݄��A�\*

loss�?�j��       �	Oʁᄘ�A�\*

loss�?a�e?       �	�n	愘�A�\*

loss�τ?�]��       �	�фꄘ�A�\*

lossx�[?i��]       �	����A�\*

lossl._?LϹ_       �	�Zv��A�\*

loss�@q?��       �	�������A�\*

loss5�?��0|       �	ob^����A�]*

loss�">?�Z       �	"�����A�]*

loss�`8?���r       �	p�V���A�]*

loss��X?�\       �	�����A�]*

lossx�+?�>�       �	��@���A�]*

loss?��&�       �	 A����A�]*

loss�b ?�oS6       �	�tC���A�]*

loss�"Y?�+^?       �	s�����A�]*

loss�\?C�IQ       �	EH7"���A�]*

lossf\A?=#       �	���&���A�]*

loss�	R?��d       �	�-���A�]*

lossj�j?�k�       �	��1���A�]*

loss-{�?���       �	�S�5���A�]*

loss��8?���       �	��k:���A�^*

loss�;�?�C^�       �	4��>���A�^*

lossՙ�?V2��       �	(�JC���A�^*

loss0��?V1�       �	���G���A�^*

loss��B?Ϳ�       �	�?L���A�^*

loss��^?��ƻ       �	���P���A�^*

loss��;?2��P       �	��-U���A�^*

lossYT?��b�       �	���[���A�^*

loss6_? ��Q       �	ɫ`���A�^*

loss���?BE&L       �	p|d���A�^*

loss��;?+vl       �	�3�h���A�^*

lossٲ�?��y       �	X�]m���A�^*

lossl�;?5M��       �	���q���A�^*

loss�S?mA�/       �	R�?v���A�_*

lossP_5?.�	�       �	Ⱥz���A�_*

loss�?�(�       �	�k+���A�_*

loss4"4?~5@X       �	�ԝ����A�_*

lossk?wJ�~       �	����A�_*

loss�9h?,�;       �	�/X����A�_*

loss.(�?^sR       �	Gtђ���A�_*

lossI^�?W߻       �	'jG����A�_*

lossFp?�       �		O�����A�_*

loss��??8y�       �	��-����A�_*

loss�7Q?�_       �	1͜����A�_*

loss�)?�3w�       �	c~����A�_*

lossK?����       �	z����A�`*

loss~�x?��8�       �	N%�����A�`*

lossMȎ?���X       �	R����A�`*

loss��?�F�;       �	q9鼅��A�`*

loss��^?c���       �	DMe����A�`*

loss#��?��e\       �	��Ņ��A�`*

loss.}E?�j�
       �	�/Gʅ��A�`*

loss�?`�        �	�'�΅��A�`*

loss��@�Ԧ*       �	��"Ӆ��A�`*

loss�P?q���       �	�ׅ��A�`*

loss��u?h�i       �	�3܅��A�`*

loss�z�?G �        �	��{����A�`*

loss��?|�;�       �	���慘�A�`*

loss2�F?�C�       �	�8녘�A�a*

loss�4j?L:       �	�ˤ�A�a*

loss��F?r�֙       �	ͱ��A�a*

loss37�?�N�v       �	�k�����A�a*

loss�1?��X       �	�U�����A�a*

loss�??3T)X       �	OZo���A�a*

loss�?z�2       �	������A�a*

loss.	?�u<|       �	t{H
���A�a*

lossD� ?E�z�       �	G����A�a*

losszv:?v�R       �	�}���A�a*

loss���?
��9       �	�#����A�a*

loss�3?�z�~       �	�� ���A�a*

loss��D?��o        �	E�n"���A�a*

loss\Qo?����       �	>�&���A�b*

loss���?6p��       �	7�U+���A�b*

lossu`�?�U%       �	&�/���A�b*

loss�ώ?��kQ       �	3�44���A�b*

loss�ɽ?c�d�       �	��8���A�b*

loss�E�?T��       �	�=���A�b*

loss���?-'�N       �	hC���A�b*

loss(�H?�h        �	���G���A�b*

loss�*l?�Ē�       �	��KL���A�b*

lossb�)?�J�       �	t��P���A�b*

loss,�3?�T       �	32;U���A�b*

loss�sb?��@`       �	�֯Y���A�b*

lossT\%?��i       �	�,+^���A�b*

loss
{?nhJ+       �	�C�b���A�c*

loss2F?U��       �	tg���A�c*

loss�E�?kq¨       �	���k���A�c*

loss>
W?m�       �	���q���A�c*

loss;L?��       �	FxIv���A�c*

loss�)�?���       �	��z���A�c*

loss �J?�h��       �	��%���A�c*

loss
Ru?n`��       �	�ܕ����A�c*

loss�Ƣ?�h��       �	������A�c*

loss�a?w2�       �	(������A�c*

loss��7?O��^       �	�<�����A�c*

loss||K?��       �	jP`����A�c*

loss�M�?��v�       �	̶˙���A�c*

loss.>k?:�@O       �	$�����A�d*

lossPx?��       �	�k�����A�d*

lossd�|?~��A       �	�����A�d*

loss�g?��z       �	�@m����A�d*

lossf�w?]]P�       �	ɰڱ���A�d*

loss8V/?*���       �	z�H����A�d*

lossb�?v�       �	�$�����A�d*

loss�R�?\�e       �	l�.����A�d*

loss�dc?�V*       �	�]�Æ��A�d*

loss��A?�[}u       �	8kȆ��A�d*

loss�Z]?�o;       �	}�cΆ��A�d*

loss&?����       �	_�҆��A�d*

loss�8�?z��o       �	"�E׆��A�e*

loss�L?vяk       �	��ۆ��A�e*

losswڕ?N��J       �	��#����A�e*

loss�	?���       �	+��䆘�A�e*

loss&�@?���       �	��醘�A�e*

lossۧ?���T       �	��v톘�A�e*

loss sS?��.       �	�L���A�e*

loss��??2�       �	�h����A�e*

loss�-F?�ϗ       �	ӆ�����A�e*

lossf�?-�(G       �	S0���A�e*

lossv�Y?��M'       �	:����A�e*

loss�?/f       �	�
���A�e*

loss�81?�с       �	������A�e*

loss�O�?LVk       �	Q3����A�f*

loss�t?���       �	Hv���A�f*

loss�B5?"[$       �	�����A�f*

lossզ?8�5F       �	U�Y ���A�f*

lossZi?]NU2       �	��$���A�f*

loss>c3?2��       �	p>+���A�f*

loss�zV?��>       �	J�/���A�f*

lossV+?�Q,       �	��3���A�f*

loss�%C?G��4       �	�Bh8���A�f*

loss��?iCK�       �	�7�<���A�f*

loss�*3?�@Ă       �	a�\A���A�f*

loss�>�?���       �	|��E���A�f*

loss$�>f�Ŵ       �	;?J���A�f*

loss�P?�V�       �	��N���A�g*

loss��G?�А1       �	T�!S���A�g*

loss޻5?�:�       �	��uY���A�g*

loss1y@8�8       �	�6�]���A�g*

loss��(?k�       �	��Xb���A�g*

loss	I�?�s�       �	d��f���A�g*

lossz�'?v�       �	�o8k���A�g*

lossF9>?��G       �	�m�o���A�g*

loss��?�Y       �	L�#t���A�g*

loss�|?�N       �	�Y�x���A�g*

loss��|?�ߋ       �	� �|���A�g*

lossnX\?�8       �	H�d����A�g*

lossb�_? �d�       �	2 �����A�g*

loss��@?c�_       �	KX5����A�h*

loss֯R?�N�       �	�y�����A�h*

loss�y??�(��       �	Q�����A�h*

loss@2?2rƝ       �	�毙���A�h*

loss5|?�$:       �	>�����A�h*

lossl��?.�       �	�W�����A�h*

lossֿN?���       �	������A�h*

loss�}?��(�       �	F|����A�h*

lossZ��?���       �	�C��A�h*

loss��_?K\�	       �	��?����A�h*

loss��E?N\�       �	�������A�h*

lossޖ�?���       �	�8�����A�h*

loss��w?���       �	���ć��A�h*

loss�a�?B�gE       �	
��ɇ��A�i*

loss���?<�*n       �	D��·��A�i*

loss��?�%z�       �	.SOӇ��A�i*

loss��_?XzTt       �	���ׇ��A�i*

loss�?ȑh�       �	�rP܇��A�i*

loss$�?b-�       �	�������A�i*

loss�ؑ?�oϜ       �	��!燘�A�i*

loss,�O?�j       �	%X�뇘�A�i*

loss�XP?ϐ�       �	�-����A�i*

losstX1?vFS�       �	Z���A�i*

loss��s?�RO[       �	y!�����A�i*

loss��?nG1       �	�!q����A�i*

loss�V�?�k=�       �	������A�j*

loss*Y?���d       �	}U���A�j*

lossR�?Ҫ�A       �	�^�
���A�j*

loss�ʂ?1N�l       �	߿7���A�j*

loss��?���       �	�E����A�j*

loss&=�?Zѷy       �	>����A�j*

lossJ�H?�zH       �	�y���A�j*

loss0�$?~�i�       �	J)�"���A�j*

loss�I ?<�Ţ       �	�Ee'���A�j*

loss�n?�n{}       �	{��+���A�j*

loss�3?I��+       �	�B0���A�j*

loss*j?6���       �	+�4���A�j*

loss:/'?�P       �	��H9���A�j*

loss�;�?v3�       �	j�=���A�k*

lossF�?K�!�       �	8�D���A�k*

lossDQ?UST       �	�<~H���A�k*

loss$�a?��       �	8��L���A�k*

loss��?��z       �	�;dQ���A�k*

lossb�T?����       �	���U���A�k*

loss��o?%�Nt       �	-TZ���A�k*

loss�R�?�F�       �	7��^���A�k*

loss,(�?��1       �	>%4c���A�k*

loss�|u?f�|/       �	�R�g���A�k*

lossB{=?����       �	t'$l���A�k*

loss��2?��0T       �	��r���A�k*

lossM?qDq6       �	��v���A�k*

loss^A?_��.       �	��r{���A�l*

loss6Z{?���       �	�B����A�l*

loss4��?��2�       �	8�\����A�l*

lossrU`?�B��       �	O�Ԉ���A�l*

loss��:?�g[i       �	��J����A�l*

lossDg?8�}       �	�O�����A�l*

lossl�?��d       �	��2����A�l*

loss��W?
o��       �	�������A�l*

loss�L?t�r       �	ߥ����A�l*

loss\��>; Jx       �	�Tw����A�l*

loss���?S�dS       �	�橈��A�l*

lossvXi?���       �	�_����A�l*

loss�D?��,       �	�ٲ���A�l*

loss��K?,�Tc       �	N�i����A�m*

lossq�?��c�       �	��ٻ���A�m*

loss�&?޲��       �	�yP����A�m*

lossby2?|���       �	_��Ĉ��A�m*

lossUB�?�B}       �	��:Ɉ��A�m*

loss�?���.       �	���ψ��A�m*

loss�@^?eo�       �	i�Ԉ��A�m*

loss+&?�g��       �	��؈��A�m*

lossD�?�>��       �	��݈��A�m*

loss��,?��_       �	
�yመ�A�m*

loss��>��ư       �	hx�刘�A�m*

loss���>\l�       �	X_ꈘ�A�m*

loss&�?=�
�       �	d���A�m*

loss&!??CA;       �	�{<��A�n*

loss�Le?:`�       �	>!�����A�n*

loss��?�!��       �	I�����A�n*

loss?`Z<       �	�Ox���A�n*

loss\C?��Ԫ       �	������A�n*

loss�Ԋ?]�#y       �	35]���A�n*

loss�?S �       �	�L����A�n*

lossf�k?�-�$       �	l�A���A�n*

loss��X?�h�M       �	е����A�n*

loss��]?l���       �	�;/���A�n*

loss�PD?J���       �	tB�!���A�n*

lossd�I?#X*�       �	��&���A�n*

loss��$?����       �	�t,���A�o*

loss�gZ?jv�       �	�*�0���A�o*

loss� ?��       �	g*S5���A�o*

loss��?�Y�N       �	/��9���A�o*

loss8hD?��v@       �	�j,>���A�o*

loss�]?od{f       �	ծ�B���A�o*

loss�6?�~       �	}=G���A�o*

loss�O?U       �	g�K���A�o*

loss�%?.oI       �	J��O���A�o*

loss�?��=m       �	�-qT���A�o*

lossF�?H�       �	,�Z���A�o*

loss�4�?��S       �	��6_���A�o*

lossZ�?W�!�       �	�Q�c���A�o*

loss��?�Z       �	W�"h���A�p*

loss�^�?H!|       �	8זl���A�p*

lossP�*?~ň�       �	rq���A�p*

loss�g?���;       �	Ԃ�u���A�p*

lossm?ό��       �	M��y���A�p*

loss��?��v�       �	��e~���A�p*

lossEC?3T=�       �	��₉��A�p*

loss�1? ��       �	��8����A�p*

loss��C?���F       �	������A�p*

loss$�?�U=�       �	�8����A�p*

loss���?�e3       �	�箖���A�p*

loss���?O��2       �	@0����A�p*

lossx�;?���q       �	J������A�p*

loss�?����       �	/�����A�q*

loss�?Iw��       �	3������A�q*

lossb;@?�o�%       �	Ad����A�q*

loss�u?�       �	�t�����A�q*

loss~�!?^��       �	��1����A�q*

lossV�;?���       �	�������A�q*

loss�<>?��b       �	��#����A�q*

lossӽ�?���       �	mƭŉ��A�q*

loss�{;?���       �	6sʉ��A�q*

loss6^.?�-{\       �	){�Ή��A�q*

lossJ�?�*�       �	��Ӊ��A�q*

loss��Y?'<�       �	�K�׉��A�q*

loss��w?��       �	�1܉��A�q*

loss�#U?�ó=       �	��|����A�r*

lossRM?Hۭ       �	���托�A�r*

lossf{E?��y�       �	I�W뉘�A�r*

loss��`?v��       �	���A�r*

lossdO=?2�H(       �	K�J��A�r*

loss�*?�"@       �	k}�����A�r*

loss8GD?��~f       �	��D����A�r*

loss�A?���       �	pw����A�r*

loss�BE?�)�       �	Ț<���A�r*

loss�U?[��f       �	�6�
���A�r*

loss���>ȡ�E       �	uV-���A�r*

lossd:T?=D�       �	�Ӕ���A�r*

loss��?��'�       �	����A�r*

lossD�"?8��(       �	Cs����A�s*

loss�M?a�8�       �	�Bv#���A�s*

loss��8?w��       �	V~(���A�s*

loss
W9?�4&       �	���,���A�s*

loss��?r��       �	�y1���A�s*

lossP�5?�{�       �	��6���A�s*

loss���>ZQN�       �	y��:���A�s*

loss ]p?-       �	�o3?���A�s*

loss^3�?L�b�       �	��E���A�s*

loss�x?v�2�       �	��J���A�s*

loss�Iy?�
�j       �	)�'O���A�s*

loss��'?�qU�       �	-$�S���A�s*

lossN�:?�fM       �	��.X���A�t*

lossW��?�Z��       �	��\���A�t*

loss��D?Oe��       �	�lFa���A�t*

lossz�?7f(�       �	0,�e���A�t*

loss,V�>��m�       �	�sj���A�t*

loss4*K?Օ�       �	k�o���A�t*

loss+C?�rr\       �	>u�u���A�t*

loss�iE?9O}H       �	B�Oz���A�t*

loss\�?l/:�       �	�;���A�t*

loss�?���       �	�Ƀ���A�t*

loss �F?lK��       �	D3^����A�t*

loss�K?7`       �	g�񌊘�A�t*

loss�I]?�M       �	������A�t*

loss�.�?��C�       �	�����A�u*

loss��?�OU$       �	2������A�u*

loss�xo?R�E�       �	��Y����A�u*

loss\B1?@�U�       �	d�𥊘�A�u*

loss4�?���#       �	q�����A�u*

loss;��?��       �	eQ����A�u*

loss��?8'��       �	�~L����A�u*

loss���?;<�       �	5�򸊘�A�u*

loss~	y?��I       �	-y�����A�u*

lossң�?�b+       �	h%J��A�u*

loss�n[?�m-       �	O��Ɗ��A�u*

lossp�?��l       �	�ˊ��A�u*

loss,�'?�Y`       �	��Њ��A�u*

lossb{?W��       �	^��֊��A�v*

loss�FJ?*�D/       �	XsOۊ��A�v*

lossN�@�K(�       �	��ߊ��A�v*

loss:z?Kٰt       �	�x䊘�A�v*

loss*�:?i]>�       �	�B銘�A�v*

loss�S?�窛       �	��튘�A�v*

loss�X?�Ӊ       �	t	b��A�v*

lossv�?I.d�       �	� ����A�v*

loss�3?�y$       �	�������A�v*

loss�7?XRT-       �	C ���A�v*

loss��?qW[�       �	����A�v*

loss �[??g�       �	������A�v*

loss+n�?����       �	�5���A�v*

loss�J?�)�       �	m�����A�w*

lossF03?��h       �	8�����A�w*

loss�pI?#e�       �	r����A�w*

loss�hj? Y|�       �	��^#���A�w*

loss��:?��Np       �	�b,(���A�w*

loss\�M?���       �	���,���A�w*

loss�|?�^��       �	�1���A�w*

loss�k?�뤸       �	��P8���A�w*

loss�?�Yh       �	u;�<���A�w*

loss�[ ?0y=L       �	dt�A���A�w*

lossf�#?�j|}       �	�3F���A�w*

loss�J?|R��       �	i��J���A�w*

loss(�\?���       �	�xO���A�w*

loss�Q�?]��c       �	m�T���A�x*

loss�6�>#��       �	)u�X���A�x*

loss&?��d       �	p�]���A�x*

loss�Z4?^�}�       �		��b���A�x*

loss�4? ��       �	g'9j���A�x*

loss|?9���       �	�CUo���A�x*

lossF"?��       �	Ht���A�x*

loss��B?F���       �	~��x���A�x*

loss��Z?�A�       �	��}���A�x*

loss�+?�
Lq       �	
,A����A�x*

loss&V?�#A�       �	Ɖ憋��A�x*

loss��'?�ߐ       �	�Ұ����A�x*

lossH:N?۬�1       �	*h����A�y*

loss"F?SSM       �	��%����A�y*

losst=z?�
�       �	��𛋘�A�y*

loss�pM?�       �	�������A�y*

lossrq-?0O2�       �	d�H����A�y*

loss(�f?OGO       �	е����A�y*

loss�;O?4i4       �	x񵮋��A�y*

loss���?O�       �	Oyz����A�y*

loss&.?!�+d       �	&B����A�y*

lossֲ?K��5       �	�|�����A�y*

lossp=?*#Y�       �	������A�y*

loss�r
?�j,m       �		NmƋ��A�y*

loss%:�?/J��       �	Y�͋��A�y*

loss��"?P[{�       �	�d�ы��A�z*

loss���?���v       �	�N�֋��A�z*

loss��(?�u�       �	��Jۋ��A�z*

loss��V?e�uc       �	÷����A�z*

loss>X?��̲       �	8��䋘�A�z*

loss��;?	�T       �	Pǜ鋘�A�z*

loss�^?|77L       �	�I�A�z*

loss�P?�]�       �	�{��A�z*

loss >@?�       �	������A�z*

loss[��?A�,I       �	Rcq����A�z*

loss�?P�WJ       �	z�B���A�z*

loss��&?s��       �	�+ ���A�z*

loss�n@�(B       �	v�����A�z*

lossz��?�_Z�       �	\Y����A�{*

loss|z8?V�e       �	��J���A�{*

loss��_?�hI       �	������A�{*

lossx�d?�Sm2       �	�����A�{*

loss��]?]h       �	��`$���A�{*

lossy?5C�       �	B&)���A�{*

loss2�w?Nז       �	��/���A�{*

loss.T5?>h��       �	x�4���A�{*

loss�:D?yy��       �	c�M9���A�{*

loss劥?�tB       �	�>���A�{*

loss�-:?�jHH       �	ƿ�B���A�{*

lossv2�??|r�       �	�_MG���A�{*

loss�xJ?.>�       �	��K���A�{*

loss��?�M�D       �	^��P���A�|*

loss�?���5       �	�yMU���A�|*

loss(_?�"A       �	Ҋ�Y���A�|*

loss��?��        �	P��`���A�|*

lossll?���h       �	�:�e���A�|*

loss&�3?�.��       �	d�xj���A�|*

loss(
M?���       �	]�2o���A�|*

loss �9?Q-��       �	���s���A�|*

losst�6?��h       �	�׮x���A�|*

loss(hI?���&       �	�d}���A�|*

loss�Q<?;}�V       �	�����A�|*

loss�N#?�*��       �	��܆���A�|*

loss��?��	       �	Ú�����A�|*

loss��_?��c~       �	��Z����A�}*

loss",??'��       �	I$����A�}*

loss�Ĕ?�T}       �	�'�����A�}*

loss k�?�=��       �	 �����A�}*

loss�W?�;Q       �	犥���A�}*

loss��
?�Ͻ       �	��e����A�}*

loss��l?� ��       �	�6<����A�}*

loss.�!?�[�       �	�"����A�}*

lossTK?�,��       �	.�)����A�}*

loss�c=?��ϋ       �	2>����A�}*

lossU?>!�       �	�.DŌ��A�}*

loss4 ?6>;4       �	�hʌ��A�}*

losst/?�?�       �	��Ό��A�~*

loss���?z�x�       �	.<�ӌ��A�~*

loss\6<?ʱ�:       �	�Z�،��A�~*

loss��K?!�A�       �	�m݌��A�~*

loss
�;?��       �	��=⌘�A�~*

loss�?ǜ��       �	 9猘�A�~*

loss r�?1���       �	)$쌘�A�~*

loss��?���       �	G������A�~*

lossZ�?����       �	�������A�~*

lossơ?UI�       �	�*u����A�~*

loss�)?9���       �	�J���A�~*

loss�,?�M��       �	����A�~*

lossDR?��y�       �	\�
���A�~*

loss�h?��zP       �	7�����A�*

lossfo?�as       �	������A�*

loss�^@����       �	�G���A�*

lossd?-1�g       �	fg����A�*

lossHY
?+8��       �	qX�"���A�*

loss
BC?���       �	nOF)���A�*

loss�?X�-       �	y�-���A�*

loss�C?��       �	(C�2���A�*

loss$ZE?G�ѿ       �	���7���A�*

loss�R^?�|�#       �	��1<���A�*

loss�?��       �	�@�@���A�*

loss��K?V��~       �	��aE���A�*

loss,�?E*��       �	�\�I���A�*

loss#?��|�       ���	�@�N���A��*

loss��5?�ǖ�       ���	��S���A��*

loss�B?��#m       ���	D�Y���A��*

lossuZ�?��f       ���	�&J^���A��*

lossHg�>�N`�       ���	ū�b���A��*

loss�l�?�/��       ���	�Rrg���A��*

loss096?�ˡ�       ���	�l���A��*

loss@��>u:�5       ���	��p���Aˀ*

loss�J=?b�OE       ���	 8Au���AՀ*

loss��/?��s       ���	�$�y���A߀*

lossbQ�?m�g(       ���	9ֈ~���A�*

loss6�t?�2�       ���	ù����A�*

loss�1{?_���       ���	O�����A��*

loss¸B?�T��       ���	7�J����A��*

loss�vD?$�Q�       ���	J蒍��A��*

loss8�B?Bn��       ���	,ԍ����A��*

loss:�?���       ���	��#����A��*

lossFa3?���r       ���	z������A��*

loss�lD?>�_�       ���	�W����A��*

loss��?[~�!       ���	
�����AÁ*

lossz�y?{��#       ���	{������A́*

loss�
?Z�F       ���	w�J����Aׁ*

loss~t?ʽH�       ���	O�׹���A�*

losst=�?���       ���	c�
����A�*

loss�l2?�Ձ^       ���	��Í��A��*

lossI·?	�q       ���	 ;�ȍ��A��*

loss�[3?qWe       ���	���͍��A��*

loss��r?�6��       ���	�uҍ��A��*

lossFuP?����       ���	�Ea׍��A��*

loss�Y?�k�       ���	�bQ܍��A��*

lossX�?�>�O       ���	=�Uፘ�A��*

loss`�d?JW       ���	`V:捘�A��*

loss|rk?��K_       ���	E5퍘�Ał*

loss&�W?�35�       ���	V ��Aς*

loss�X?�p�{       ���	�����Aق*

loss`�.?�+��       ���	�k�����A�*

loss��?5�_�       ���	�����A�*

loss��X?\I{       ���	K@���A��*

lossVF�?��;       ���	,}���A��*

lossn7\?���/       ���	#�����A��*

loss��"?��B       ���	����A��*

loss�'?�pg       ���	�����A��*

lossx�w?_��l       ���	�-"���A��*

loss��?I�5�       ���	#!'���A��*

loss�R(?/�we       ���	2�$,���A��*

loss�R5?�Y�       ���	<f:1���Aǃ*

loss��{?�3��       ���	�@6���Aу*

loss(H?�U�       ���	!WQ;���Aۃ*

loss��#?�@(�       ���	�\�@���A�*

loss�0?� )�       ���	�i�E���A�*

lossl�m?-g�w       ���	ǸtJ���A��*

loss �!?|�5�       ���	�͍O���A��*

loss"A?e~H       ���	�*W���A��*

lossT6	?+qQ       ���	-!\���A��*

lossZ(?��f�       ���	��Ka���A��*

loss�n1?�m�
       ���	�UYf���A��*

loss�Y?=��       ���	��k���A��*

loss�?�>d�%       ���	���p���A��*

loss�I?�Cw�       ���	=|v���AɄ*

loss.�f?Z�p�       ���	r4{���Aӄ*

loss���?M��       ���	��N����A݄*

loss��"?�ޖ�       ���	B�t����A�*

loss���>����       ���	� ����A�*

loss<�Z?��       ���	4�����A��*

loss��?/�Y�       ���	��D����A��*

lossY?��%       ���	��t����A��*

loss�%?.=��       ���	t�š���A��*

lossn9?o�ߡ       ���	��覎��A��*

lossdG6?�pK�       ���	�>,����A��*

lossH?��9a       ���	��O����A��*

loss��q?y�/�       ���	H������A��*

loss�w\?7n��       ���	_C�����A˅*

loss��;?F�       ���	+�Î��AՅ*

lossΎO?&�       ���	
ITȎ��A߅*

loss:I8?�       ���	�t�͎��A�*

loss䝥?����       ���	v,ӎ��A�*

loss��[?�f��       ���	k�َ��A��*

lossZ�K? `        ���	;Umގ��A��*

loss1��?�(\i       ���	[�㎘�A��*

loss�?��?�       ���	���莘�A��*

loss��C?�@6)       ���	ō�A��*

loss8��>��w       ���	Kw���A��*

loss��>� ��       ���	������A��*

lossW��?k�       ���	��E����AÆ*

lossh�Q?3K��       ���	1E����A͆*

losst4�>�Q�       ���	@�	���A׆*

loss~--??��       ���	�����A�*

lossU�l?}.�       ���	�m����A�*

loss��+?،)       ���	������A��*

loss��?)Ц�       ���	h[� ���A��*

loss�"A?@�u       ���	F�'���A��*

loss�_B?&�h�       ���	��,���A��*

loss�61?d]*�       ���	�:4���A��*

loss��'?�A�       ���	��D9���A��*

lossL�?HL�~       ���	V{>���A��*

losszJ?y�H�       ���	śC���A��*

loss�g?'���       ���	W�H���AŇ*

loss��?bk�       ���	 ��M���Aχ*

loss� ?���Z       ���	��R���Aه*

loss8�]?-Z!$       ���	�6
X���A�*

loss.�G?]��	       ���	�2*^���A�*

loss��5? s�       ���	ͯc���A��*

loss��?�j�I       ���	p��k���A��*

losspkR?��4       ���	ϡ�p���A��*

loss@�?>�8�       ���	��nv���A��*

lossz�8?e[�$       ���	���{���A��*

loss�k?���       ���	t�����A��*

loss��?Χ=�       ���	W>f����A��*

loss��A?EO<       ���	^�����A��*

loss�?i��       ���	�kΐ���Aǈ*

loss��v?�I�       ���	*����Aш*

loss8�q?�k�D       ���	eqC����Aۈ*

loss�1p?�K`�       ���	3m�����A�*

loss�8�>7d       ���	�>����A�*

loss��[?��@�       ���	^����A��*

loss��,?!)�L       ���	��ݳ���A��*

loss���>��5       ���	�B����A��*

loss>M,?���       ���	:橾���A��*

loss4>c?p       ���	�}2ď��A��*

loss���>��E       ���	ܞuɏ��A��*

loss��	?i �       ���	���Ώ��A��*

loss��?1�^d       ���	��ӏ��A��*

loss@X�>�}�       ���	x_Dۏ��Aɉ*

losshz?�ܭy       ���	�������AӉ*

lossW$g?|��        ���	��变�A݉*

lossP�b?3���       ���	B%kꏘ�A�*

lossN.?].(V       ���	��\�A�*

loss��S?�̦]       ���	�3��A��*

loss��?��w       ���	�C����A��*

loss�-?��.       ���	������A��*

loss�?���~       ���	B�����A��*

loss|3?,e2�       ���	������A��*

lossXy3?���A       ���	%?����A��*

loss��_?=�o       ���	p�����A��*

loss��*?��N       ���	�ܔ���A��*

loss>xZ?�q9"       ���	v���Aˊ*

loss�?��3       ���	F"���AՊ*

lossނ�?@��       ���	i'���Aߊ*

loss�=)?�B�G       ���	�4�+���A�*

loss^.?��>       ���	J��0���A�*

loss|�S?�n1�       ���	.��5���A��*

loss
b*?�|�1       ���	��:���A��*

loss��y?~?       ���	aQA���A��*

loss�?k��       ���	:�OF���A��*

loss�5L?�>�S       ���	��)K���A��*

loss��@?��܍       ���	%��O���A��*

lossR�!?���b       ���	e��T���A��*

loss46??*�{       ���	�;�Y���AË*

loss�PR?Sl��       ���	d?�^���A͋*

loss`d?z��)       ���	�N[c���A׋*

loss��>����       ���	%<=h���A�*

lossN2#?T�j       ���	�m���A�*

loss�'?�~)~       ���	j��s���A��*

loss޿%?���       ���	_
�x���A��*

loss0�>�Q��       ���	-|�}���A��*

loss��?��=�       ���	��T����A��*

lossح?z��       ���	@�.����A��*

loss(�q?����       ���	������A��*

loss��??��m       ���	�����A��*

loss|�?@6?       ���	b�)����A��*

loss�{�?����       ���	�ߘ����AŌ*

loss�s?8A�       ���	Kv!����Aό*

loss.::?�Y|       ���	�3g����Aٌ*

lossZ43?Xy'       ���	s�y����A�*

loss#�?�Ƿ       ���	z�-����A�*

loss�?�	       ���	�嶐��A��*

losshn?��%3       ���	�$�����A��*

lossT�l?��}i       ���	��_����A��*

loss��S?Y�R�       ���	�R&Ő��A��*

loss��o?ι��       ���	m<�ɐ��A��*

loss*?��z�       ���	T�ΐ��A��*

lossv�7?/\x�       ���	�aӐ��A��*

lossP��>�l�       ���	H�&ڐ��A��*

loss�o?�eD�       ���	�B�ސ��AǍ*

loss?�?���       ���	=��㐘�Aэ*

loss05?���       ���	�*�萘�Aۍ*

loss@��>��,�       ���	��퐘�A�*

loss�q9?ۗa�       ���	8I���A�*

lossX�6?ۨ�       ���	V�����A��*

loss�?�U�       ���	b�)����A��*

lossj�	?� �       ���	�X���A��*

losspwU?�&�       ���	[�����A��*

loss�F�?@�.x       ���	�����A��*

loss|�O?Rߑ
       ���	Gʜ���A��*

loss�mB?,Cj       ���	�����A��*

loss<��>��AH       ���	�r���A��*

loss^M?��!6       ���	�B~#���AɎ*

loss"�M?ԫ��       ���	��(���Aӎ*

loss��?5�Ջ       ���	��.���Aݎ*

lossnq.?o�-x       ���	dxb3���A�*

loss�|?'_       ���	��y8���A�*

loss��#?d�c       ���	X�p=���A��*

loss4n
?�[Y�       ���	�wuD���A��*

loss}�?�U�       ���	m�]I���A��*

loss�6�?��g	       ���	O�2N���A��*

lossJ++?R~�       ���	�CS���A��*

loss��?�ժG       ���	�f0X���A��*

loss�?�4a�       ���	�Zp]���A��*

loss�x7?6���       ���	�C~b���A��*

loss"�3?�Pv
       ���	��vg���Aˏ*

loss���?Wx�E       ���	���l���AՏ*

loss�5?�	�       ���	��$r���Aߏ*

loss�T?�K%�       ���	ۆ�y���A�*

loss�I?/;��       ���	K� ���A�*

lossƦC?)�bx       ���	%둄���A��*

loss@��>����       ���	eo�����A��*

lossb.�?Z?       ���	h^�����A��*

loss �X?~9�,       ���	��\����A��*

lossp�*?mk��       ���	0�8����A��*

loss�&?8��       ���	|�W����A��*

loss �D?���       ���	�(����A��*

loss"�g?9x�u       ���	먹����AÐ*

loss�Y?U]t       ���	�w<����A͐*

loss��?6�i       ���	������Aא*

loss4w2?�W�       ���	��&Ñ��A�*

loss�7�>�5�       ���	|��ɑ��A�*

lossd�G?n�_A       ���	�z�ϑ��A��*

loss��k?��)       ���	P�֑��A��*

lossȹJ?       ���	�[kۑ��A��*

loss�k	?c,��       ���	��ᑘ�A��*

lossP�n?�3T�       ���	�U\摘�A��*

loss>?o�|F       ���	a�쑘�A��*

lossp=?��.9       ���	+h:����A��*

loss�61?ؤ�i       ���	��3����A��*

lossr\W?~��       ���	d� ���Aő*

loss��i?;���       ���	�4j���Aϑ*

loss�.�?�4�D       ���		�I���Aّ*

loss�D?M(@       ���	\W`���A�*

loss֒T?�s�       ���	��T���A�*

loss|�>��       ���	�����A��*

lossƶ$?��       ���	Y5�$���A��*

loss�f�>F�       ���	}\2+���A��*

loss�=8?�&�        ���	w�N4���A��*

loss��?�Iv       ���	b��:���A��*

lossf�z?M�om       ���	ʋ�@���A��*

loss�"?���       ���	A)G���A��*

loss?w       ���	��"M���A��*

loss�,?��GV       ���	�{�R���Aǒ*

loss1J?�F�L       ���	�ӜX���Aђ*

lossx�v?vʭ�       ���	Da^���Aے*

loss�
C?�%��       ���	#(d���A�*

loss��?��       ���	i��i���A�*

loss8UJ?ņV,       ���	ۢ�r���A��*

loss5�>j�       ���	��|y���A��*

lossHMM?̄o-       ���	=����A��*

loss�^M?��i�       ���	�(u����A��*

loss�N.?5�K�       ���	ni6����A��*

loss�_K?�d��       ���	"n󓒘�A��*

loss�e?*       ���	I.e����A��*

loss��?���$       ���	&������A��*

losshR?��+       ���	�������Aɓ*

lossh�7?W�       ���	x~�����Aӓ*

loss��A?��       ���	� ����Aݓ*

loss)?x�p       ���	5_콒��A�*

loss�+?=��       ���	�Œ��A�*

lossr�N?�n�x       ���	j�'͒��A��*

loss��+?�       ���	o�Ԓ��A��*

losst��?��k�       ���	�W�ے��A��*

loss��?DC�Y       ���	kfL⒘�A��*

loss E�>p{��       ���	���钘�A��*

loss10?��&       ���	�����A��*

lossP�G?H,�       ���	؛#����A��*

lossFQ?��e       ���	E�(���A��*

losshg?l�R       ���	��o���A˔*

loss�Y8?�z[       ���	A+$���AՔ*

loss̞�>�y�4       ���	�L����Aߔ*

lossp��>���       ���	��\���A�*

loss\�!?~GD�       ���	��!���A�*

loss��??СV�       ���	���(���A��*

loss��6?U�ʣ       ���	M0/���A��*

loss|��>|��Z       ���	'/D7���A��*

loss�K#?�YG9       ���	���=���A��*

lossHe�>50��       ���	 G���A��*

loss~�S?�ɱ       ���	�VM���A��*

loss�6?l/u       ���	]�S���A��*

loss��>.��       ���	��[���AÕ*

loss(�.?����       ���	k��b���A͕*

loss�$?����       ���	���h���Aו*

loss�b?�Rh       ���	�|o���A�*

loss�$|?�+�       ���	{�v���A�*

loss�L?��I       ���	�|���A��*

loss�t<?q-��       ���	�t����A��*

loss�@}0�`       ���	9�݋���A��*

loss?F}��       ���	k�?����A��*

lossN�?�N,�       ���	�=�����A��*

loss�Vl?��2m       ���	J' ����A��*

loss�|6?�1m       ���	�������A��*

loss��>�c��       ���	������A��*

loss��>
��       ���	�]b����AŖ*

loss
; ?���z       ���	�A����Aϖ*

loss��0?:��       ���	��=����Aٖ*

loss�a�>��S       ���	�fƓ��A�*

loss�7G?���       ���	��Г��A�*

loss2� ?v��       ���	��֓��A��*

loss-%?egu�       ���	��Pܓ��A��*

lossd�[?����       ���	Gt�ⓘ�A��*

loss�)?d��       ���	�MB铘�A��*

lossJ�?8�@       ���	�z��A��*

loss��I?G��       ���	��F����A��*

loss(��>����       ���	������A��*

lossz].?�
�       ���	?n���A��*

lossDC?0�o�       ���	�{e	���AǗ*

loss&��?LvV�       ���	W����Aї*

lossq?U�&�       ���	������Aۗ*

loss44�?	r�       ���	^i����A�*

loss(�R?h�       ���	. @&���A�*

lossJ�l?>h�$       ���	Xr�,���A��*

loss�?�6�v       ���	V��3���A��*

loss��<?��-�       ���	|D.:���A��*

loss�ȃ?�9"       ���	�n�@���A��*

loss<�L?���6       ���	��G���A��*

loss`�>�@�       ���	pBZO���A��*

loss�`@?˄J       ���	�Y���A��*

loss2@O?���       ���	X��_���A��*

loss��n?�<       ���	
ef���Aɘ*

loss��?�f�       ���	�3�l���AӘ*

lossd��>�|��       ���	��s���Aݘ*

loss�34?is       ���	��yz���A�*

lossX?� z       ���	]�݀���A�*

lossn|<?rhp       ���	��6����A��*

lossp��>R��       ���	�5�����A��*

loss�8?��9�       ���	�UȔ���A��*

loss"�7?XMy       ���	�U����A��*

lossD��>C�R       ���	Ãf����A��*

lossp��>��d       ���	��v����A��*

loss,p�>w�v       ���	t~y����A��*

lossl�0?���       ���	������A��*

loss�4?��)�       ���	�������A˙*

lossN�8?��       ���	�X0ǔ��Aՙ*

loss"=~?�<M�       ���	%�͔��Aߙ*

loss�D;?Q�(B       ���	/��Ԕ��A�*

loss�N?�	U       ���	!@۔��A�*

loss`N%?9��S       ���	�W䔘�A��*

loss�vy?,P�       ���	^딘�A��*

lossU?idG       ���	A���A��*

loss�հ>V��       ���	������A��*

loss�s?���       ���	�4Z����A��*

loss��?W��[       ���	�a���A��*

loss��?�t�       ���	0�����A��*

loss��?����       ���	�#	���AÚ*

lossF�?3�?       ���	������A͚*

loss&�5?}��X       ���	J{� ���Aך*

loss<��>�?X       ���	�]*���A�*

loss:�'?C��       ���	�D1���A�*

lossP��>6O'       ���	�4�7���A��*

lossCA?�_E�       ���	��S>���A��*

lossx�6?GO��       ���	���D���A��*

lossT�$?[��       ���	xD�J���A��*

losshy�>hg��       ���	B@GQ���A��*

losst�?|��