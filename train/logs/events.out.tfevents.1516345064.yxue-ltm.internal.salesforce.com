       гK"	   ║eШ╓Abrain.Event:2з╜№а]Д     VЯ Т	G╔║eШ╓A"╨И
z
imgPlaceholder*
dtype0*&
shape:         АР*1
_output_shapes
:         АР
h
labelPlaceholder*
dtype0*
shape:         *'
_output_shapes
:         
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
╣
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
д
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
ж
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  А?*
_output_shapes
: 
Н
Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w
У
5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Б
1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
┴
coarse/conv1/conv1-w
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ё
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Х
coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Ь
&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
й
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
┌
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Й
coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
▀
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*/
_output_shapes
:         @H*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
┼
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*/
_output_shapes
:         @H*
T0*
data_formatNHWC

coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*/
_output_shapes
:         @H
═
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
╕
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  А?*
_output_shapes
: 
л
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
╗
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
й
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╒
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
Щ
%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
│
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
░
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
╜
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
В
%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
з
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
И
3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*/
_output_shapes
:         @H *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*/
_output_shapes
:         @H *
T0*
data_formatNHWC
Э
&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  А?*
_output_shapes
: 
л
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
▌
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
М
coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ь
&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
й
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
┌
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
Й
coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
к
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
╖
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
Ў
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
Ю
 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
▒
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  А?*
_output_shapes
: 
┐
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
Е
&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
к
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
Л
!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
║
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*J
_output_shapes8
6:         @H :         @H 
╙
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
╤
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
╡
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%oГ:*G
_output_shapes5
3:         @H : : : : *
T0*
is_training(*
data_formatNHWC
╝
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*J
_output_shapes8
6:         @H :         @H 
╒
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
╙
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
с
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
щ
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
▌
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*G
_output_shapes5
3:         @H : : : : *
T0*
is_training( *
data_formatNHWC
╔
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*1
_output_shapes
:         @H : *
T0*
N
║
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
_output_shapes

: : *
T0*
N
║
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
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
┤
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
й
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
╪
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
╥
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ф
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
│
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
р
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
┌
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
Ё
(coarse/coarse/conv2-bn/AssignMovingAvg_1	AssignSubcoarse/conv2-bn/moving_variance,coarse/coarse/conv2-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking( *
T0*
_output_shapes
: 
}
coarse/coarse/conv2-reluRelu!coarse/coarse/conv2-bn/cond/Merge*
T0*/
_output_shapes
:         @H 
╞
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:          $ *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
╕
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  А?*
_output_shapes
: 
л
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
╗
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
й
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╒
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
Щ
%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
│
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
░
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
╜
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
В
%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
з
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
К
3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:          $@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*/
_output_shapes
:          $@*
T0*
data_formatNHWC
Э
&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  А?*
_output_shapes
:@
л
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
▌
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
М
coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ь
&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
й
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
┌
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
Й
coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
к
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
╖
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
Ў
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
Ю
 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
▒
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  А?*
_output_shapes
:@
┐
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
Е
&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
к
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
Л
!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
║
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:          $@:          $@
╙
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
╤
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
╡
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%oГ:*G
_output_shapes5
3:          $@:@:@:@:@*
T0*
is_training(*
data_formatNHWC
╝
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:          $@:          $@
╒
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
╙
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
с
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
щ
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
▌
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*G
_output_shapes5
3:          $@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
╔
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*1
_output_shapes
:          $@: *
T0*
N
║
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
_output_shapes

:@: *
T0*
N
║
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
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
┤
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
й
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
╪
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
╥
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ф
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
│
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
р
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
┌
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
Ё
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
:          $@
╚
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:         @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @   А   *
_output_shapes
:
╕
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  А?*
_output_shapes
: 
м
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@А*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
╝
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
к
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╫
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
Ъ
%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
┤
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
▓
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
┐
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Г
%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
и
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
Н
3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:         А*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ю
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*0
_output_shapes
:         А*
T0*
data_formatNHWC
Я
&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*  А?*
_output_shapes	
:А
н
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
▐
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Н
coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
Ю
&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
л
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
█
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
К
coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
м
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueBА*    *
_output_shapes	
:А
╣
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
ў
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Я
 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
│
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueBА*  А?*
_output_shapes	
:А
┴
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 
Ж
&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
л
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
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
Л
!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
╝
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:         А:         А
╒
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:А:А
╙
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:А:А
║
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%oГ:*L
_output_shapes:
8:         А:А:А:А:А*
T0*
is_training(*
data_formatNHWC
╛
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:         А:         А
╫
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:А:А
╒
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:А:А
у
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
:А:А
ы
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
:А:А
т
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*L
_output_shapes:
8:         А:А:А:А:А*
T0*
is_training( *
data_formatNHWC
╩
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*2
_output_shapes 
:         А: *
T0*
N
╗
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
_output_shapes
	:А: *
T0*
N
╗
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
_output_shapes
	:А: *
T0*
N
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
┤
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
к
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
┘
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
╙
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
х
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:А
┤
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
с
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
█
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
ё
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:А
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:         А
╔
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:         	А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*%
valueB"	   	   А      *
_output_shapes
:
╕
@coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
valueB
 *  А?*
_output_shapes
: 
н
Kcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/shape*(
_output_shapes
:		АА*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w
╜
?coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
л
;coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normalAdd?coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mul@coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
┘
coarse/conv5-conv/conv5-conv-w
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
Ы
%coarse/conv5-conv/conv5-conv-w/AssignAssigncoarse/conv5-conv/conv5-conv-w;coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
╡
#coarse/conv5-conv/conv5-conv-w/readIdentitycoarse/conv5-conv/conv5-conv-w*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
▓
0coarse/conv5-conv/conv5-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
┐
coarse/conv5-conv/conv5-conv-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Г
%coarse/conv5-conv/conv5-conv-b/AssignAssigncoarse/conv5-conv/conv5-conv-b0coarse/conv5-conv/conv5-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
и
#coarse/conv5-conv/conv5-conv-b/readIdentitycoarse/conv5-conv/conv5-conv-b*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
Н
3coarse/coarse/conv5-conv/conv5-conv/conv5-conv-convConv2Dcoarse/coarse/MaxPool_2#coarse/conv5-conv/conv5-conv-w/read*0
_output_shapes
:         	А*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ю
7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_addBiasAdd3coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv#coarse/conv5-conv/conv5-conv-b/read*0
_output_shapes
:         	А*
T0*
data_formatNHWC
Я
&coarse/conv5-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*  А?*
_output_shapes	
:А
н
coarse/conv5-bn/gamma
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
▐
coarse/conv5-bn/gamma/AssignAssigncoarse/conv5-bn/gamma&coarse/conv5-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Н
coarse/conv5-bn/gamma/readIdentitycoarse/conv5-bn/gamma*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
Ю
&coarse/conv5-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
л
coarse/conv5-bn/beta
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
█
coarse/conv5-bn/beta/AssignAssigncoarse/conv5-bn/beta&coarse/conv5-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
К
coarse/conv5-bn/beta/readIdentitycoarse/conv5-bn/beta*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
м
-coarse/conv5-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
valueBА*    *
_output_shapes	
:А
╣
coarse/conv5-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
shared_name 
ў
"coarse/conv5-bn/moving_mean/AssignAssigncoarse/conv5-bn/moving_mean-coarse/conv5-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Я
 coarse/conv5-bn/moving_mean/readIdentitycoarse/conv5-bn/moving_mean*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
│
0coarse/conv5-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
valueBА*  А?*
_output_shapes	
:А
┴
coarse/conv5-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
shared_name 
Ж
&coarse/conv5-bn/moving_variance/AssignAssigncoarse/conv5-bn/moving_variance0coarse/conv5-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
л
$coarse/conv5-bn/moving_variance/readIdentitycoarse/conv5-bn/moving_variance*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
m
"coarse/coarse/conv5-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv5-bn/cond/switch_tIdentity$coarse/coarse/conv5-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv5-bn/cond/switch_fIdentity"coarse/coarse/conv5-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv5-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:
Л
!coarse/coarse/conv5-bn/cond/ConstConst%^coarse/coarse/conv5-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv5-bn/cond/Const_1Const%^coarse/coarse/conv5-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
╝
1coarse/coarse/conv5-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add*
T0*L
_output_shapes:
8:         	А:         	А
╒
3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id*(
_class
loc:@coarse/conv5-bn/gamma*
T0*"
_output_shapes
:А:А
╙
3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id*'
_class
loc:@coarse/conv5-bn/beta*
T0*"
_output_shapes
:А:А
║
*coarse/coarse/conv5-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv5-bn/cond/Const#coarse/coarse/conv5-bn/cond/Const_1*
epsilon%oГ:*L
_output_shapes:
8:         	А:А:А:А:А*
T0*
is_training(*
data_formatNHWC
╛
3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add*
T0*L
_output_shapes:
8:         	А:         	А
╫
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id*(
_class
loc:@coarse/conv5-bn/gamma*
T0*"
_output_shapes
:А:А
╒
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id*'
_class
loc:@coarse/conv5-bn/beta*
T0*"
_output_shapes
:А:А
у
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv5-bn/moving_mean/read#coarse/coarse/conv5-bn/cond/pred_id*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*"
_output_shapes
:А:А
ы
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv5-bn/moving_variance/read#coarse/coarse/conv5-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*"
_output_shapes
:А:А
т
,coarse/coarse/conv5-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*L
_output_shapes:
8:         	А:А:А:А:А*
T0*
is_training( *
data_formatNHWC
╩
!coarse/coarse/conv5-bn/cond/MergeMerge,coarse/coarse/conv5-bn/cond/FusedBatchNorm_1*coarse/coarse/conv5-bn/cond/FusedBatchNorm*2
_output_shapes 
:         	А: *
T0*
N
╗
#coarse/coarse/conv5-bn/cond/Merge_1Merge.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv5-bn/cond/FusedBatchNorm:1*
_output_shapes
	:А: *
T0*
N
╗
#coarse/coarse/conv5-bn/cond/Merge_2Merge.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv5-bn/cond/FusedBatchNorm:2*
_output_shapes
	:А: *
T0*
N
l
'coarse/coarse/conv5-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv5-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
!coarse/coarse/conv5-bn/ExpandDims
ExpandDims'coarse/coarse/conv5-bn/ExpandDims/input%coarse/coarse/conv5-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv5-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv5-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
╢
#coarse/coarse/conv5-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv5-bn/ExpandDims_1/input'coarse/coarse/conv5-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv5-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
П
coarse/coarse/conv5-bn/ReshapeReshapeis_training$coarse/coarse/conv5-bn/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
┤
coarse/coarse/conv5-bn/SelectSelectcoarse/coarse/conv5-bn/Reshape!coarse/coarse/conv5-bn/ExpandDims#coarse/coarse/conv5-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv5-bn/SqueezeSqueezecoarse/coarse/conv5-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
к
+coarse/coarse/conv5-bn/AssignMovingAvg/readIdentitycoarse/conv5-bn/moving_mean*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
┘
*coarse/coarse/conv5-bn/AssignMovingAvg/SubSub+coarse/coarse/conv5-bn/AssignMovingAvg/read#coarse/coarse/conv5-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
╙
*coarse/coarse/conv5-bn/AssignMovingAvg/MulMul*coarse/coarse/conv5-bn/AssignMovingAvg/Subcoarse/coarse/conv5-bn/Squeeze*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
х
&coarse/coarse/conv5-bn/AssignMovingAvg	AssignSubcoarse/conv5-bn/moving_mean*coarse/coarse/conv5-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:А
┤
-coarse/coarse/conv5-bn/AssignMovingAvg_1/readIdentitycoarse/conv5-bn/moving_variance*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
с
,coarse/coarse/conv5-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv5-bn/AssignMovingAvg_1/read#coarse/coarse/conv5-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
█
,coarse/coarse/conv5-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv5-bn/AssignMovingAvg_1/Subcoarse/coarse/conv5-bn/Squeeze*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
ё
(coarse/coarse/conv5-bn/AssignMovingAvg_1	AssignSubcoarse/conv5-bn/moving_variance,coarse/coarse/conv5-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:А
~
coarse/coarse/conv5-reluRelu!coarse/coarse/conv5-bn/cond/Merge*
T0*0
_output_shapes
:         	А
╔
coarse/coarse/MaxPool_3MaxPoolcoarse/coarse/conv5-relu*0
_output_shapes
:         А*
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
valueB"     ╠ *
_output_shapes
:
Ш
coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_3coarse/coarse/Reshape/shape*)
_output_shapes
:         АШ*
T0*
Tshape0
й
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB" ╠    *
_output_shapes
:
Ь
2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 
Ю
4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  А?*
_output_shapes
: 
№
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
:АША*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
■
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
ь
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
п
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
▄
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
Д
coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
Ц
"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
г
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
╦
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
о
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
н
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*(
_output_shapes
:         А*
T0*
data_formatNHWC
й
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"      *
_output_shapes
:
Ь
2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 
Ю
4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  А?*
_output_shapes
: 
·
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	А*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
№
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
ъ
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
л
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
┌
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
В
coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
Ф
"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
б
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
╩
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
║
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
м
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*'
_output_shapes
:         *
T0*
data_formatNHWC
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:         
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
:         
J
add/yConst*
dtype0*
valueB
 *╠╝М+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:         
C
SqrtSqrtadd*
T0*'
_output_shapes
:         
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
в
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
д
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
┬
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:
Р
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
н
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
п
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
о
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
у
gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
╞
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0
х
gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
╩
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
	keep_dims( *

Tidx0
▀
gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
▓
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
░
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
л
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
н
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
й
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
М
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
╖
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:         
╧
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
л
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
н
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
┤
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:         
н
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  А?*
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
:         
Б
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:         
б
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
▒
gradients/Pow_grad/Greater/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:         
д
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:         
▒
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:         
и
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:         
Г
gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:         
Ж
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:         
е
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
М
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
╖
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
┌
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:         
╧
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
╩
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
п
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╕
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
Ы
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
╖
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
┌
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:         
р
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:         
╡
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0*
data_formatNHWC
И
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
й
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:         
╙
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
∙
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         А
 
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	А
Г
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
┴
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:         А
╛
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	А
╨
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
_output_shapes	
:А*
T0*
data_formatNHWC
в
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
▌
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:         А
╘
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
·
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:         АШ
Ї
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:АША
Г
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
┬
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:         АШ
└
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
:АША
╤
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
є
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*0
_output_shapes
:         А*
T0*
Tshape0
п
2gradients/coarse/coarse/MaxPool_3_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv5-relucoarse/coarse/MaxPool_3,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:         	А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┼
0gradients/coarse/coarse/conv5-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_3_grad/MaxPoolGradcoarse/coarse/conv5-relu*
T0*0
_output_shapes
:         	А
╖
:gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv5-relu_grad/ReluGrad#coarse/coarse/conv5-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:         	А:         	А
╓
Agradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_grad
╒
Igradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*0
_output_shapes
:         	А
┘
Kgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*0
_output_shapes
:         	А
╟
gradients/zeros_like	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Ь
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*L
_output_shapes:
8:         	А:А:А:А:А*
T0*
is_training( *
data_formatNHWC
ї
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Э
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         	А
М
Vgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
М
Vgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Д
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv5-bn/cond/FusedBatchNorm:3,coarse/coarse/conv5-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*F
_output_shapes4
2:         	А:А:А: : *
T0*
is_training(*
data_formatNHWC
ё
Jgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Х
Rgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         	А
Д
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Д
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Б
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Я
gradients/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         	А:         	А
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
к
gradients/zeros/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*0
_output_shapes
:         	А
В
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*2
_output_shapes 
:         	А: *
T0*
N
┌
gradients/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_1/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
_output_shapes
	:А: *
T0*
N
┘
gradients/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_2/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
_output_shapes
	:А: *
T0*
N
б
gradients/Switch_3Switch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         	А:         	А
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
А
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:         	А
А
Jgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*2
_output_shapes 
:         	А: *
T0*
N
┌
gradients/Switch_4Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_4/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes	
:А
я
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
_output_shapes
	:А: *
T0*
N
┘
gradients/Switch_5Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_5/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes	
:А
я
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
_output_shapes
	:А: *
T0*
N
╒
gradients/AddNAddNLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         	А*
N
о
Rgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
_output_shapes	
:А*
T0*
data_formatNHWC
Х
Wgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGrad
ё
_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         	А
и
agradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
╚
gradients/AddN_1AddNNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:А*
N
╚
gradients/AddN_2AddNNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:А*
N
е
Igradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_2#coarse/conv5-conv/conv5-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ъ
Vgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeN#coarse/conv5-conv/conv5-conv-w/read_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
т
Wgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_2Kgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilter
╗
[gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInput*
T0*0
_output_shapes
:         	А
╖
]gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:		АА
▐
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2[gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency*0
_output_shapes
:         А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┼
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:         А
╖
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:         А:         А
╓
Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
╒
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:         А
┘
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:         А
╔
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╩
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╩
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Ь
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*L
_output_shapes:
8:         А:А:А:А:А*
T0*
is_training( *
data_formatNHWC
ї
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Э
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         А
М
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
М
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Д
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*F
_output_shapes4
2:         А:А:А: : *
T0*
is_training(*
data_formatNHWC
ё
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Х
Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         А
Д
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Д
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Б
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
б
gradients/Switch_6Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         А:         А
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_6/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
А
gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*0
_output_shapes
:         А
Д
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*2
_output_shapes 
:         А: *
T0*
N
┌
gradients/Switch_7Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_8Shapegradients/Switch_7:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_7/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
_output_shapes
	:А: *
T0*
N
┘
gradients/Switch_8Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_8/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
_output_shapes
	:А: *
T0*
N
б
gradients/Switch_9Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         А:         А
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
Б
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*0
_output_shapes
:         А
А
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*2
_output_shapes 
:         А: *
T0*
N
█
gradients/Switch_10Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_11Shapegradients/Switch_10*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_10/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
n
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes	
:А
Ё
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
_output_shapes
	:А: *
T0*
N
┌
gradients/Switch_11Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_12Shapegradients/Switch_11*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_11/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
n
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*
_output_shapes	
:А
Ё
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
_output_shapes
	:А: *
T0*
N
╫
gradients/AddN_3AddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         А*
N
░
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes	
:А*
T0*
data_formatNHWC
Ч
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
є
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         А
и
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
╚
gradients/AddN_4AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:А*
N
╚
gradients/AddN_5AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:А*
N
е
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ъ
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
т
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:         @
╢
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@А
▌
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:          $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
─
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:          $@
╡
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:          $@:          $@
╓
Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
╘
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:          $@
╪
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:          $@
╔
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
Ч
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*G
_output_shapes5
3:          $@:@:@:@:@*
T0*
is_training( *
data_formatNHWC
ї
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Ь
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:          $@
Л
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Л
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
╟
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
Б
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*C
_output_shapes1
/:          $@:@:@: : *
T0*
is_training(*
data_formatNHWC
ё
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Ф
Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:          $@
Г
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Г
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Б
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
а
gradients/Switch_12Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:          $@:          $@
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*/
_output_shapes
:          $@
Д
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*1
_output_shapes
:          $@: *
T0*
N
┘
gradients/Switch_13Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
g
gradients/Shape_14Shapegradients/Switch_13:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_13/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
:@
є
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
_output_shapes

:@: *
T0*
N
╪
gradients/Switch_14Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_14/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
:@
є
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
_output_shapes

:@: *
T0*
N
а
gradients/Switch_15Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:          $@:          $@
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*/
_output_shapes
:          $@
А
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*1
_output_shapes
:          $@: *
T0*
N
┘
gradients/Switch_16Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_17Shapegradients/Switch_16*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_16/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
:@
я
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
_output_shapes

:@: *
T0*
N
╪
gradients/Switch_17Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_18Shapegradients/Switch_17*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_17/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
:@
я
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
_output_shapes

:@: *
T0*
N
╓
gradients/AddN_6AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:          $@*
N
п
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
_output_shapes
:@*
T0*
data_formatNHWC
Ч
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
Є
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:          $@
з
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
╟
gradients/AddN_7AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@*
N
╟
gradients/AddN_8AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@*
N
г
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ъ
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
р
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:          $ 
╡
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
┘
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:         @H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┬
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*/
_output_shapes
:         @H 
╡
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:         @H :         @H 
╓
Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
╘
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*/
_output_shapes
:         @H 
╪
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*/
_output_shapes
:         @H 
╔
gradients/zeros_like_24	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_25	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_26	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_27	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
Ч
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*G
_output_shapes5
3:         @H : : : : *
T0*
is_training( *
data_formatNHWC
ї
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Ь
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:         @H 
Л
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Л
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
╟
gradients/zeros_like_28	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_29	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_30	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_31	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
Б
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*C
_output_shapes1
/:         @H : : : : *
T0*
is_training(*
data_formatNHWC
ё
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Ф
Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:         @H 
Г
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Г
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
а
gradients/Switch_18Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:         @H :         @H 
g
gradients/Shape_19Shapegradients/Switch_18:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_18/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*
T0*/
_output_shapes
:         @H 
Д
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*1
_output_shapes
:         @H : *
T0*
N
┘
gradients/Switch_19Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_20Shapegradients/Switch_19:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_19/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
T0*
_output_shapes
: 
є
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
_output_shapes

: : *
T0*
N
╪
gradients/Switch_20Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_21Shapegradients/Switch_20:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_20/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*
_output_shapes
: 
є
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
_output_shapes

: : *
T0*
N
а
gradients/Switch_21Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:         @H :         @H 
e
gradients/Shape_22Shapegradients/Switch_21*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_21/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*
T0*/
_output_shapes
:         @H 
А
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_21*1
_output_shapes
:         @H : *
T0*
N
┘
gradients/Switch_22Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_23Shapegradients/Switch_22*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_22/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*
T0*
_output_shapes
: 
я
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_22*
_output_shapes

: : *
T0*
N
╪
gradients/Switch_23Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_24Shapegradients/Switch_23*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_23/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
T0*
_output_shapes
: 
я
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_23*
_output_shapes

: : *
T0*
N
╓
gradients/AddN_9AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:         @H *
N
п
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_9*
_output_shapes
: *
T0*
data_formatNHWC
Ч
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_9S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
Є
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_9X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:         @H 
з
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
╚
gradients/AddN_10AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: *
N
╚
gradients/AddN_11AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: *
N
б
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
ъ
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:         @H
╡
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
у
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*/
_output_shapes
:         @H
╗
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
Ф
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
╬
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*/
_output_shapes
:         @H
ы
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
°
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0* 
_output_shapes
::*
N
│
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
б
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▒
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter
А
Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:         АР
∙
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
З
beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 
Ш
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
╖
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
З
beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *w╛?*
_output_shapes
: 
Ш
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
╖
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
╣
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
╞
coarse/conv1/conv1-w/Adam
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ї
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Я
coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
╗
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
╚
coarse/conv1/conv1-w/Adam_1
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
√
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
г
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
б
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
о
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
щ
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
У
coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
г
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
░
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
я
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Ч
 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
═
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
┌
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
Э
*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
╜
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╧
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
▄
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
г
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
┴
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╡
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
┬
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
С
*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
▒
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
╖
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
─
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
Ч
,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
╡
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
г
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
░
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
э
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Ц
coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
е
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
▓
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
є
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Ъ
!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
б
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
о
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
щ
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
У
coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
г
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
░
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
я
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
Ч
 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
═
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
┌
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
Э
*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
╜
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╧
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
▄
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
г
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
┴
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╡
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
┬
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
С
*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
▒
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
╖
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
─
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
Ч
,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
╡
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
г
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
░
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
э
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Ц
coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
е
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
▓
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
є
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Ъ
!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
б
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
о
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
щ
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
У
coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
г
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
░
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
я
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
Ч
 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
╧
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@А*    *'
_output_shapes
:@А
▄
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
Ю
*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
╛
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╤
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@А*    *'
_output_shapes
:@А
▐
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
д
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
┬
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╖
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
─
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Т
*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
▓
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
╣
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
╞
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Ш
,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
╢
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
е
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
ю
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ч
coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
з
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*    *
_output_shapes	
:А
┤
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
Ї
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ы
!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
г
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
░
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
ъ
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ф
coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
е
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
Ё
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ш
 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
╤
5coarse/conv5-conv/conv5-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*'
valueB		АА*    *(
_output_shapes
:		АА
▐
#coarse/conv5-conv/conv5-conv-w/Adam
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
Я
*coarse/conv5-conv/conv5-conv-w/Adam/AssignAssign#coarse/conv5-conv/conv5-conv-w/Adam5coarse/conv5-conv/conv5-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
┐
(coarse/conv5-conv/conv5-conv-w/Adam/readIdentity#coarse/conv5-conv/conv5-conv-w/Adam*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
╙
7coarse/conv5-conv/conv5-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*'
valueB		АА*    *(
_output_shapes
:		АА
р
%coarse/conv5-conv/conv5-conv-w/Adam_1
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
е
,coarse/conv5-conv/conv5-conv-w/Adam_1/AssignAssign%coarse/conv5-conv/conv5-conv-w/Adam_17coarse/conv5-conv/conv5-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
├
*coarse/conv5-conv/conv5-conv-w/Adam_1/readIdentity%coarse/conv5-conv/conv5-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
╖
5coarse/conv5-conv/conv5-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
─
#coarse/conv5-conv/conv5-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Т
*coarse/conv5-conv/conv5-conv-b/Adam/AssignAssign#coarse/conv5-conv/conv5-conv-b/Adam5coarse/conv5-conv/conv5-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
▓
(coarse/conv5-conv/conv5-conv-b/Adam/readIdentity#coarse/conv5-conv/conv5-conv-b/Adam*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
╣
7coarse/conv5-conv/conv5-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
╞
%coarse/conv5-conv/conv5-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Ш
,coarse/conv5-conv/conv5-conv-b/Adam_1/AssignAssign%coarse/conv5-conv/conv5-conv-b/Adam_17coarse/conv5-conv/conv5-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
╢
*coarse/conv5-conv/conv5-conv-b/Adam_1/readIdentity%coarse/conv5-conv/conv5-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
е
,coarse/conv5-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv5-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
ю
!coarse/conv5-bn/gamma/Adam/AssignAssigncoarse/conv5-bn/gamma/Adam,coarse/conv5-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ч
coarse/conv5-bn/gamma/Adam/readIdentitycoarse/conv5-bn/gamma/Adam*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
з
.coarse/conv5-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*    *
_output_shapes	
:А
┤
coarse/conv5-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
Ї
#coarse/conv5-bn/gamma/Adam_1/AssignAssigncoarse/conv5-bn/gamma/Adam_1.coarse/conv5-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ы
!coarse/conv5-bn/gamma/Adam_1/readIdentitycoarse/conv5-bn/gamma/Adam_1*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
г
+coarse/conv5-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
░
coarse/conv5-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
ъ
 coarse/conv5-bn/beta/Adam/AssignAssigncoarse/conv5-bn/beta/Adam+coarse/conv5-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ф
coarse/conv5-bn/beta/Adam/readIdentitycoarse/conv5-bn/beta/Adam*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
е
-coarse/conv5-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv5-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
Ё
"coarse/conv5-bn/beta/Adam_1/AssignAssigncoarse/conv5-bn/beta/Adam_1-coarse/conv5-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ш
 coarse/conv5-bn/beta/Adam_1/readIdentitycoarse/conv5-bn/beta/Adam_1*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
з
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueBАША*    *!
_output_shapes
:АША
┤
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
р
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
О
coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
й
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueBАША*    *!
_output_shapes
:АША
╢
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ц
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
Т
coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
Ы
'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
и
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
┌
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
И
coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
Э
)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
к
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
р
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
М
coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
г
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	А*    *
_output_shapes
:	А
░
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
▐
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
М
coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
е
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	А*    *
_output_shapes
:	А
▓
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
ф
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
Р
coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
Щ
'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
ж
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
┘
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
З
coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Ы
)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
и
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
▀
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
Л
coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Я

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Я

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w╛?*
_output_shapes
: 
б
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w╠+2*
_output_shapes
: 
д
*Adam/update_coarse/conv1/conv1-w/ApplyAdam	ApplyAdamcoarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonNgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-w*
use_locking( *
T0*&
_output_shapes
:
Ь
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
х
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
▌
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
р
+Adam/update_coarse/conv2-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_nesterov( *(
_class
loc:@coarse/conv2-bn/gamma*
use_locking( *
T0*
_output_shapes
: 
█
*Adam/update_coarse/conv2-bn/beta/ApplyAdam	ApplyAdamcoarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
use_nesterov( *'
_class
loc:@coarse/conv2-bn/beta*
use_locking( *
T0*
_output_shapes
: 
х
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
▌
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
▀
+Adam/update_coarse/conv3-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_nesterov( *(
_class
loc:@coarse/conv3-bn/gamma*
use_locking( *
T0*
_output_shapes
:@
┌
*Adam/update_coarse/conv3-bn/beta/ApplyAdam	ApplyAdamcoarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *'
_class
loc:@coarse/conv3-bn/beta*
use_locking( *
T0*
_output_shapes
:@
ц
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@А
▐
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:А
р
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:А
█
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:А
ч
4Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam	ApplyAdamcoarse/conv5-conv/conv5-conv-w#coarse/conv5-conv/conv5-conv-w/Adam%coarse/conv5-conv/conv5-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking( *
T0*(
_output_shapes
:		АА
▐
4Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam	ApplyAdamcoarse/conv5-conv/conv5-conv-b#coarse/conv5-conv/conv5-conv-b/Adam%coarse/conv5-conv/conv5-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking( *
T0*
_output_shapes	
:А
р
+Adam/update_coarse/conv5-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv5-bn/gammacoarse/conv5-bn/gamma/Adamcoarse/conv5-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv5-bn/gamma*
use_locking( *
T0*
_output_shapes	
:А
█
*Adam/update_coarse/conv5-bn/beta/ApplyAdam	ApplyAdamcoarse/conv5-bn/betacoarse/conv5-bn/beta/Adamcoarse/conv5-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv5-bn/beta*
use_locking( *
T0*
_output_shapes	
:А
Д
&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
:АША
Г
&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:А
В
&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	А
В
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
Щ	
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
Я
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ы	

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
г
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ь
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Ї
save/SaveV2/tensor_namesConst*
dtype0*з
valueЭBЪLBbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/conv5-bn/betaBcoarse/conv5-bn/beta/AdamBcoarse/conv5-bn/beta/Adam_1Bcoarse/conv5-bn/gammaBcoarse/conv5-bn/gamma/AdamBcoarse/conv5-bn/gamma/Adam_1Bcoarse/conv5-bn/moving_meanBcoarse/conv5-bn/moving_varianceBcoarse/conv5-conv/conv5-conv-bB#coarse/conv5-conv/conv5-conv-b/AdamB%coarse/conv5-conv/conv5-conv-b/Adam_1Bcoarse/conv5-conv/conv5-conv-wB#coarse/conv5-conv/conv5-conv-w/AdamB%coarse/conv5-conv/conv5-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:L
■
save/SaveV2/shape_and_slicesConst*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
╟
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powercoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1coarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1coarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1coarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1coarse/conv2-bn/moving_meancoarse/conv2-bn/moving_variancecoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1coarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1coarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1coarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1coarse/conv3-bn/moving_meancoarse/conv3-bn/moving_variancecoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1coarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1coarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1coarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1coarse/conv4-bn/moving_meancoarse/conv4-bn/moving_variancecoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1coarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1coarse/conv5-bn/betacoarse/conv5-bn/beta/Adamcoarse/conv5-bn/beta/Adam_1coarse/conv5-bn/gammacoarse/conv5-bn/gamma/Adamcoarse/conv5-bn/gamma/Adam_1coarse/conv5-bn/moving_meancoarse/conv5-bn/moving_variancecoarse/conv5-conv/conv5-conv-b#coarse/conv5-conv/conv5-conv-b/Adam%coarse/conv5-conv/conv5-conv-b/Adam_1coarse/conv5-conv/conv5-conv-w#coarse/conv5-conv/conv5-conv-w/Adam%coarse/conv5-conv/conv5-conv-w/Adam_1coarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1coarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1coarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1coarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1*Z
dtypesP
N2L
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
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
е
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
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
й
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
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
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
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Б
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
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╜
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
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_5Assigncoarse/conv1/conv1-wsave/RestoreV2_5*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
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
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Б
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
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
╔
save/Assign_7Assigncoarse/conv1/conv1-w/Adam_1save/RestoreV2_7*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
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
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
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
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
В
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
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
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
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Б
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
Щ
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Г
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
Щ
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
┴
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
В
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
Щ
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
╞
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
Ж
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
Щ
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
╬
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
Е
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
Щ
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
К
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
Щ
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
╤
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
М
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
Щ
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
Е
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
Щ
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
╪
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
К
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
Щ
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
▌
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
М
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
Щ
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
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
Щ
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
А
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
Щ
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
╜
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
В
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
Щ
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
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
Щ
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Б
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
Щ
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Г
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
Щ
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
┴
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
В
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
Щ
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
╞
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
Ж
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
Щ
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
╬
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
Е
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
Щ
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
К
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
Щ
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
╤
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
М
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
Щ
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
Е
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
Щ
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
╪
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
К
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
Щ
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
▌
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
М
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
Щ
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
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
Щ
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
╣
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
А
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
Щ
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
В
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
Щ
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
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
Щ
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Б
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
Щ
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Г
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
Щ
save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
В
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
Щ
save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Ж
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
Щ
save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
╧
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
Е
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
Щ
save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
═
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
К
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
Щ
save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
М
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
Щ
save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
╘
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
Е
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
Щ
save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
┘
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
К
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
Щ
save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
▐
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
М
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
Щ
save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
р
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
{
save/RestoreV2_50/tensor_namesConst*
dtype0*)
value BBcoarse/conv5-bn/beta*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
╣
save/Assign_50Assigncoarse/conv5-bn/betasave/RestoreV2_50*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
А
save/RestoreV2_51/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv5-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_51Assigncoarse/conv5-bn/beta/Adamsave/RestoreV2_51*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
В
save/RestoreV2_52/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv5-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_52Assigncoarse/conv5-bn/beta/Adam_1save/RestoreV2_52*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
|
save/RestoreV2_53/tensor_namesConst*
dtype0**
value!BBcoarse/conv5-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_53Assigncoarse/conv5-bn/gammasave/RestoreV2_53*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Б
save/RestoreV2_54/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv5-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_54Assigncoarse/conv5-bn/gamma/Adamsave/RestoreV2_54*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Г
save/RestoreV2_55/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv5-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_55Assigncoarse/conv5-bn/gamma/Adam_1save/RestoreV2_55*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
В
save/RestoreV2_56/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv5-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_56Assigncoarse/conv5-bn/moving_meansave/RestoreV2_56*
validate_shape(*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Ж
save/RestoreV2_57/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv5-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
╧
save/Assign_57Assigncoarse/conv5-bn/moving_variancesave/RestoreV2_57*
validate_shape(*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
Е
save/RestoreV2_58/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv5-conv/conv5-conv-b*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
═
save/Assign_58Assigncoarse/conv5-conv/conv5-conv-bsave/RestoreV2_58*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
К
save/RestoreV2_59/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv5-conv/conv5-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_59Assign#coarse/conv5-conv/conv5-conv-b/Adamsave/RestoreV2_59*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
М
save/RestoreV2_60/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv5-conv/conv5-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_60/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
╘
save/Assign_60Assign%coarse/conv5-conv/conv5-conv-b/Adam_1save/RestoreV2_60*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
Е
save/RestoreV2_61/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv5-conv/conv5-conv-w*
_output_shapes
:
k
"save/RestoreV2_61/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
┌
save/Assign_61Assigncoarse/conv5-conv/conv5-conv-wsave/RestoreV2_61*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
К
save/RestoreV2_62/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv5-conv/conv5-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_62/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_62	RestoreV2
save/Constsave/RestoreV2_62/tensor_names"save/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
save/Assign_62Assign#coarse/conv5-conv/conv5-conv-w/Adamsave/RestoreV2_62*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
М
save/RestoreV2_63/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv5-conv/conv5-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_63/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_63	RestoreV2
save/Constsave/RestoreV2_63/tensor_names"save/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
с
save/Assign_63Assign%coarse/conv5-conv/conv5-conv-w/Adam_1save/RestoreV2_63*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
w
save/RestoreV2_64/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-b*
_output_shapes
:
k
"save/RestoreV2_64/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_64	RestoreV2
save/Constsave/RestoreV2_64/tensor_names"save/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
▒
save/Assign_64Assigncoarse/fc1/fc1-bsave/RestoreV2_64*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
|
save/RestoreV2_65/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-b/Adam*
_output_shapes
:
k
"save/RestoreV2_65/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_65	RestoreV2
save/Constsave/RestoreV2_65/tensor_names"save/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
save/Assign_65Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_65*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
~
save/RestoreV2_66/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_66/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_66	RestoreV2
save/Constsave/RestoreV2_66/tensor_names"save/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_66Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_66*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
w
save/RestoreV2_67/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-w*
_output_shapes
:
k
"save/RestoreV2_67/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_67	RestoreV2
save/Constsave/RestoreV2_67/tensor_names"save/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
╖
save/Assign_67Assigncoarse/fc1/fc1-wsave/RestoreV2_67*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
|
save/RestoreV2_68/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-w/Adam*
_output_shapes
:
k
"save/RestoreV2_68/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_68	RestoreV2
save/Constsave/RestoreV2_68/tensor_names"save/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
╝
save/Assign_68Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_68*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
~
save/RestoreV2_69/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_69/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_69	RestoreV2
save/Constsave/RestoreV2_69/tensor_names"save/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_69Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_69*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
w
save/RestoreV2_70/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-b*
_output_shapes
:
k
"save/RestoreV2_70/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_70	RestoreV2
save/Constsave/RestoreV2_70/tensor_names"save/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
░
save/Assign_70Assigncoarse/fc2/fc2-bsave/RestoreV2_70*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
|
save/RestoreV2_71/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-b/Adam*
_output_shapes
:
k
"save/RestoreV2_71/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_71	RestoreV2
save/Constsave/RestoreV2_71/tensor_names"save/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
╡
save/Assign_71Assigncoarse/fc2/fc2-b/Adamsave/RestoreV2_71*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
~
save/RestoreV2_72/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_72/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_72	RestoreV2
save/Constsave/RestoreV2_72/tensor_names"save/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
╖
save/Assign_72Assigncoarse/fc2/fc2-b/Adam_1save/RestoreV2_72*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
w
save/RestoreV2_73/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-w*
_output_shapes
:
k
"save/RestoreV2_73/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_73	RestoreV2
save/Constsave/RestoreV2_73/tensor_names"save/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
╡
save/Assign_73Assigncoarse/fc2/fc2-wsave/RestoreV2_73*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
|
save/RestoreV2_74/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-w/Adam*
_output_shapes
:
k
"save/RestoreV2_74/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_74	RestoreV2
save/Constsave/RestoreV2_74/tensor_names"save/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_74Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_74*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
~
save/RestoreV2_75/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_75/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_75	RestoreV2
save/Constsave/RestoreV2_75/tensor_names"save/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
╝
save/Assign_75Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_75*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
Ш

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"э╟ф╛¤     Юкиq	╡3i║eШ╓AJ▒√
╒,│,
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
2	АР
ы
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignSub
ref"TА

value"T

output_ref"TА"
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
╚
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
ю
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
э
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
И
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
epsilonfloat%╖╤8"
data_formatstringNHWC"
is_trainingbool(
░
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
epsilonfloat%╖╤8"
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
╙
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
ы
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
2	Р
К
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
2	Р
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
К
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Й
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
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514╨И
z
imgPlaceholder*
dtype0*&
shape:         АР*1
_output_shapes
:         АР
h
labelPlaceholder*
dtype0*
shape:         *'
_output_shapes
:         
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
╣
7coarse/conv1/conv1-w/Initializer/truncated_normal/shapeConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB"            *
_output_shapes
:
д
6coarse/conv1/conv1-w/Initializer/truncated_normal/meanConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *    *
_output_shapes
: 
ж
8coarse/conv1/conv1-w/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*
valueB
 *  А?*
_output_shapes
: 
Н
Acoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7coarse/conv1/conv1-w/Initializer/truncated_normal/shape*&
_output_shapes
:*
dtype0*
seed2 *

seed *
T0*'
_class
loc:@coarse/conv1/conv1-w
У
5coarse/conv1/conv1-w/Initializer/truncated_normal/mulMulAcoarse/conv1/conv1-w/Initializer/truncated_normal/TruncatedNormal8coarse/conv1/conv1-w/Initializer/truncated_normal/stddev*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Б
1coarse/conv1/conv1-w/Initializer/truncated_normalAdd5coarse/conv1/conv1-w/Initializer/truncated_normal/mul6coarse/conv1/conv1-w/Initializer/truncated_normal/mean*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
┴
coarse/conv1/conv1-w
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ё
coarse/conv1/conv1-w/AssignAssigncoarse/conv1/conv1-w1coarse/conv1/conv1-w/Initializer/truncated_normal*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Х
coarse/conv1/conv1-w/readIdentitycoarse/conv1/conv1-w*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
Ь
&coarse/conv1/conv1-b/Initializer/ConstConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
й
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
┌
coarse/conv1/conv1-b/AssignAssigncoarse/conv1/conv1-b&coarse/conv1/conv1-b/Initializer/Const*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Й
coarse/conv1/conv1-b/readIdentitycoarse/conv1/conv1-b*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
▀
$coarse/coarse/conv1/conv1/conv1-convConv2Dimgcoarse/conv1/conv1-w/read*/
_output_shapes
:         @H*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
┼
(coarse/coarse/conv1/conv1/conv1-biad_addBiasAdd$coarse/coarse/conv1/conv1/conv1-convcoarse/conv1/conv1-b/read*
data_formatNHWC*
T0*/
_output_shapes
:         @H

coarse/coarse/relu1Relu(coarse/coarse/conv1/conv1/conv1-biad_add*
T0*/
_output_shapes
:         @H
═
Acoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB"             *
_output_shapes
:
╕
@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
valueB
 *  А?*
_output_shapes
: 
л
Kcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w
╗
?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
й
;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normalAdd?coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mul@coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╒
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
Щ
%coarse/conv2-conv/conv2-conv-w/AssignAssigncoarse/conv2-conv/conv2-conv-w;coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
│
#coarse/conv2-conv/conv2-conv-w/readIdentitycoarse/conv2-conv/conv2-conv-w*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
░
0coarse/conv2-conv/conv2-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
╜
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
В
%coarse/conv2-conv/conv2-conv-b/AssignAssigncoarse/conv2-conv/conv2-conv-b0coarse/conv2-conv/conv2-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
з
#coarse/conv2-conv/conv2-conv-b/readIdentitycoarse/conv2-conv/conv2-conv-b*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
И
3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-convConv2Dcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read*/
_output_shapes
:         @H *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_addBiasAdd3coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv#coarse/conv2-conv/conv2-conv-b/read*
data_formatNHWC*
T0*/
_output_shapes
:         @H 
Э
&coarse/conv2-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *  А?*
_output_shapes
: 
л
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
▌
coarse/conv2-bn/gamma/AssignAssigncoarse/conv2-bn/gamma&coarse/conv2-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
М
coarse/conv2-bn/gamma/readIdentitycoarse/conv2-bn/gamma*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
Ь
&coarse/conv2-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
й
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
┌
coarse/conv2-bn/beta/AssignAssigncoarse/conv2-bn/beta&coarse/conv2-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
Й
coarse/conv2-bn/beta/readIdentitycoarse/conv2-bn/beta*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
к
-coarse/conv2-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
valueB *    *
_output_shapes
: 
╖
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
Ў
"coarse/conv2-bn/moving_mean/AssignAssigncoarse/conv2-bn/moving_mean-coarse/conv2-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
Ю
 coarse/conv2-bn/moving_mean/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
▒
0coarse/conv2-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
valueB *  А?*
_output_shapes
: 
┐
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
Е
&coarse/conv2-bn/moving_variance/AssignAssigncoarse/conv2-bn/moving_variance0coarse/conv2-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
к
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
Л
!coarse/coarse/conv2-bn/cond/ConstConst%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv2-bn/cond/Const_1Const%^coarse/coarse/conv2-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
║
1coarse/coarse/conv2-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*J
_output_shapes8
6:         @H :         @H 
╙
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
╤
3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
╡
*coarse/coarse/conv2-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv2-bn/cond/Const#coarse/coarse/conv2-bn/cond/Const_1*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*G
_output_shapes5
3:         @H : : : : 
╝
3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add*
T0*J
_output_shapes8
6:         @H :         @H 
╒
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id*(
_class
loc:@coarse/conv2-bn/gamma*
T0* 
_output_shapes
: : 
╙
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id*'
_class
loc:@coarse/conv2-bn/beta*
T0* 
_output_shapes
: : 
с
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv2-bn/moving_mean/read#coarse/coarse/conv2-bn/cond/pred_id*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0* 
_output_shapes
: : 
щ
5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv2-bn/moving_variance/read#coarse/coarse/conv2-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0* 
_output_shapes
: : 
▌
,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:         @H : : : : 
╔
!coarse/coarse/conv2-bn/cond/MergeMerge,coarse/coarse/conv2-bn/cond/FusedBatchNorm_1*coarse/coarse/conv2-bn/cond/FusedBatchNorm*
N*
T0*1
_output_shapes
:         @H : 
║
#coarse/coarse/conv2-bn/cond/Merge_1Merge.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

: : 
║
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
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv2-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv2-bn/ReshapeReshapeis_training$coarse/coarse/conv2-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
┤
coarse/coarse/conv2-bn/SelectSelectcoarse/coarse/conv2-bn/Reshape!coarse/coarse/conv2-bn/ExpandDims#coarse/coarse/conv2-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv2-bn/SqueezeSqueezecoarse/coarse/conv2-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
й
+coarse/coarse/conv2-bn/AssignMovingAvg/readIdentitycoarse/conv2-bn/moving_mean*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
╪
*coarse/coarse/conv2-bn/AssignMovingAvg/SubSub+coarse/coarse/conv2-bn/AssignMovingAvg/read#coarse/coarse/conv2-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
╥
*coarse/coarse/conv2-bn/AssignMovingAvg/MulMul*coarse/coarse/conv2-bn/AssignMovingAvg/Subcoarse/coarse/conv2-bn/Squeeze*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
T0*
_output_shapes
: 
ф
&coarse/coarse/conv2-bn/AssignMovingAvg	AssignSubcoarse/conv2-bn/moving_mean*coarse/coarse/conv2-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking( *
T0*
_output_shapes
: 
│
-coarse/coarse/conv2-bn/AssignMovingAvg_1/readIdentitycoarse/conv2-bn/moving_variance*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
р
,coarse/coarse/conv2-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv2-bn/AssignMovingAvg_1/read#coarse/coarse/conv2-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
┌
,coarse/coarse/conv2-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv2-bn/AssignMovingAvg_1/Subcoarse/coarse/conv2-bn/Squeeze*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
T0*
_output_shapes
: 
Ё
(coarse/coarse/conv2-bn/AssignMovingAvg_1	AssignSubcoarse/conv2-bn/moving_variance,coarse/coarse/conv2-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking( *
T0*
_output_shapes
: 
}
coarse/coarse/conv2-reluRelu!coarse/coarse/conv2-bn/cond/Merge*
T0*/
_output_shapes
:         @H 
╞
coarse/coarse/MaxPoolMaxPoolcoarse/coarse/conv2-relu*/
_output_shapes
:          $ *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB"          @   *
_output_shapes
:
╕
@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
valueB
 *  А?*
_output_shapes
: 
л
Kcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w
╗
?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
й
;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normalAdd?coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mul@coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╒
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
Щ
%coarse/conv3-conv/conv3-conv-w/AssignAssigncoarse/conv3-conv/conv3-conv-w;coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
│
#coarse/conv3-conv/conv3-conv-w/readIdentitycoarse/conv3-conv/conv3-conv-w*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
░
0coarse/conv3-conv/conv3-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
╜
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
В
%coarse/conv3-conv/conv3-conv-b/AssignAssigncoarse/conv3-conv/conv3-conv-b0coarse/conv3-conv/conv3-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
з
#coarse/conv3-conv/conv3-conv-b/readIdentitycoarse/conv3-conv/conv3-conv-b*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
К
3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-convConv2Dcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read*/
_output_shapes
:          $@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_addBiasAdd3coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv#coarse/conv3-conv/conv3-conv-b/read*
data_formatNHWC*
T0*/
_output_shapes
:          $@
Э
&coarse/conv3-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*  А?*
_output_shapes
:@
л
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
▌
coarse/conv3-bn/gamma/AssignAssigncoarse/conv3-bn/gamma&coarse/conv3-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
М
coarse/conv3-bn/gamma/readIdentitycoarse/conv3-bn/gamma*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
Ь
&coarse/conv3-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
й
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
┌
coarse/conv3-bn/beta/AssignAssigncoarse/conv3-bn/beta&coarse/conv3-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
Й
coarse/conv3-bn/beta/readIdentitycoarse/conv3-bn/beta*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
к
-coarse/conv3-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
valueB@*    *
_output_shapes
:@
╖
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
Ў
"coarse/conv3-bn/moving_mean/AssignAssigncoarse/conv3-bn/moving_mean-coarse/conv3-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
Ю
 coarse/conv3-bn/moving_mean/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
▒
0coarse/conv3-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
valueB@*  А?*
_output_shapes
:@
┐
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
Е
&coarse/conv3-bn/moving_variance/AssignAssigncoarse/conv3-bn/moving_variance0coarse/conv3-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
к
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
Л
!coarse/coarse/conv3-bn/cond/ConstConst%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv3-bn/cond/Const_1Const%^coarse/coarse/conv3-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
║
1coarse/coarse/conv3-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:          $@:          $@
╙
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
╤
3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
╡
*coarse/coarse/conv3-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv3-bn/cond/Const#coarse/coarse/conv3-bn/cond/Const_1*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*G
_output_shapes5
3:          $@:@:@:@:@
╝
3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add*
T0*J
_output_shapes8
6:          $@:          $@
╒
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id*(
_class
loc:@coarse/conv3-bn/gamma*
T0* 
_output_shapes
:@:@
╙
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id*'
_class
loc:@coarse/conv3-bn/beta*
T0* 
_output_shapes
:@:@
с
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv3-bn/moving_mean/read#coarse/coarse/conv3-bn/cond/pred_id*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0* 
_output_shapes
:@:@
щ
5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv3-bn/moving_variance/read#coarse/coarse/conv3-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0* 
_output_shapes
:@:@
▌
,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:          $@:@:@:@:@
╔
!coarse/coarse/conv3-bn/cond/MergeMerge,coarse/coarse/conv3-bn/cond/FusedBatchNorm_1*coarse/coarse/conv3-bn/cond/FusedBatchNorm*
N*
T0*1
_output_shapes
:          $@: 
║
#coarse/coarse/conv3-bn/cond/Merge_1Merge.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes

:@: 
║
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
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv3-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv3-bn/ReshapeReshapeis_training$coarse/coarse/conv3-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
┤
coarse/coarse/conv3-bn/SelectSelectcoarse/coarse/conv3-bn/Reshape!coarse/coarse/conv3-bn/ExpandDims#coarse/coarse/conv3-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv3-bn/SqueezeSqueezecoarse/coarse/conv3-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
й
+coarse/coarse/conv3-bn/AssignMovingAvg/readIdentitycoarse/conv3-bn/moving_mean*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
╪
*coarse/coarse/conv3-bn/AssignMovingAvg/SubSub+coarse/coarse/conv3-bn/AssignMovingAvg/read#coarse/coarse/conv3-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
╥
*coarse/coarse/conv3-bn/AssignMovingAvg/MulMul*coarse/coarse/conv3-bn/AssignMovingAvg/Subcoarse/coarse/conv3-bn/Squeeze*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
T0*
_output_shapes
:@
ф
&coarse/coarse/conv3-bn/AssignMovingAvg	AssignSubcoarse/conv3-bn/moving_mean*coarse/coarse/conv3-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking( *
T0*
_output_shapes
:@
│
-coarse/coarse/conv3-bn/AssignMovingAvg_1/readIdentitycoarse/conv3-bn/moving_variance*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
р
,coarse/coarse/conv3-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv3-bn/AssignMovingAvg_1/read#coarse/coarse/conv3-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
┌
,coarse/coarse/conv3-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv3-bn/AssignMovingAvg_1/Subcoarse/coarse/conv3-bn/Squeeze*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
T0*
_output_shapes
:@
Ё
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
:          $@
╚
coarse/coarse/MaxPool_1MaxPoolcoarse/coarse/conv3-relu*/
_output_shapes
:         @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*%
valueB"      @   А   *
_output_shapes
:
╕
@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
valueB
 *  А?*
_output_shapes
: 
м
Kcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/shape*'
_output_shapes
:@А*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w
╝
?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
к
;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normalAdd?coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mul@coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╫
coarse/conv4-conv/conv4-conv-w
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
Ъ
%coarse/conv4-conv/conv4-conv-w/AssignAssigncoarse/conv4-conv/conv4-conv-w;coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
┤
#coarse/conv4-conv/conv4-conv-w/readIdentitycoarse/conv4-conv/conv4-conv-w*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
▓
0coarse/conv4-conv/conv4-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
┐
coarse/conv4-conv/conv4-conv-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Г
%coarse/conv4-conv/conv4-conv-b/AssignAssigncoarse/conv4-conv/conv4-conv-b0coarse/conv4-conv/conv4-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
и
#coarse/conv4-conv/conv4-conv-b/readIdentitycoarse/conv4-conv/conv4-conv-b*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
Н
3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-convConv2Dcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read*0
_output_shapes
:         А*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ю
7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_addBiasAdd3coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv#coarse/conv4-conv/conv4-conv-b/read*
data_formatNHWC*
T0*0
_output_shapes
:         А
Я
&coarse/conv4-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*  А?*
_output_shapes	
:А
н
coarse/conv4-bn/gamma
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
▐
coarse/conv4-bn/gamma/AssignAssigncoarse/conv4-bn/gamma&coarse/conv4-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Н
coarse/conv4-bn/gamma/readIdentitycoarse/conv4-bn/gamma*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
Ю
&coarse/conv4-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
л
coarse/conv4-bn/beta
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
█
coarse/conv4-bn/beta/AssignAssigncoarse/conv4-bn/beta&coarse/conv4-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
К
coarse/conv4-bn/beta/readIdentitycoarse/conv4-bn/beta*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
м
-coarse/conv4-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
valueBА*    *
_output_shapes	
:А
╣
coarse/conv4-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
shared_name 
ў
"coarse/conv4-bn/moving_mean/AssignAssigncoarse/conv4-bn/moving_mean-coarse/conv4-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Я
 coarse/conv4-bn/moving_mean/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
│
0coarse/conv4-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
valueBА*  А?*
_output_shapes	
:А
┴
coarse/conv4-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
shared_name 
Ж
&coarse/conv4-bn/moving_variance/AssignAssigncoarse/conv4-bn/moving_variance0coarse/conv4-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
л
$coarse/conv4-bn/moving_variance/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
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
Л
!coarse/coarse/conv4-bn/cond/ConstConst%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv4-bn/cond/Const_1Const%^coarse/coarse/conv4-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
╝
1coarse/coarse/conv4-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:         А:         А
╒
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:А:А
╙
3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:А:А
║
*coarse/coarse/conv4-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv4-bn/cond/Const#coarse/coarse/conv4-bn/cond/Const_1*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*L
_output_shapes:
8:         А:А:А:А:А
╛
3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add*
T0*L
_output_shapes:
8:         А:         А
╫
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id*(
_class
loc:@coarse/conv4-bn/gamma*
T0*"
_output_shapes
:А:А
╒
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id*'
_class
loc:@coarse/conv4-bn/beta*
T0*"
_output_shapes
:А:А
у
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv4-bn/moving_mean/read#coarse/coarse/conv4-bn/cond/pred_id*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*"
_output_shapes
:А:А
ы
5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv4-bn/moving_variance/read#coarse/coarse/conv4-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*"
_output_shapes
:А:А
т
,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:         А:А:А:А:А
╩
!coarse/coarse/conv4-bn/cond/MergeMerge,coarse/coarse/conv4-bn/cond/FusedBatchNorm_1*coarse/coarse/conv4-bn/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:         А: 
╗
#coarse/coarse/conv4-bn/cond/Merge_1Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:А: 
╗
#coarse/coarse/conv4-bn/cond/Merge_2Merge.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes
	:А: 
l
'coarse/coarse/conv4-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv4-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
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
╢
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
П
coarse/coarse/conv4-bn/ReshapeReshapeis_training$coarse/coarse/conv4-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
┤
coarse/coarse/conv4-bn/SelectSelectcoarse/coarse/conv4-bn/Reshape!coarse/coarse/conv4-bn/ExpandDims#coarse/coarse/conv4-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv4-bn/SqueezeSqueezecoarse/coarse/conv4-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
к
+coarse/coarse/conv4-bn/AssignMovingAvg/readIdentitycoarse/conv4-bn/moving_mean*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
┘
*coarse/coarse/conv4-bn/AssignMovingAvg/SubSub+coarse/coarse/conv4-bn/AssignMovingAvg/read#coarse/coarse/conv4-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
╙
*coarse/coarse/conv4-bn/AssignMovingAvg/MulMul*coarse/coarse/conv4-bn/AssignMovingAvg/Subcoarse/coarse/conv4-bn/Squeeze*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
T0*
_output_shapes	
:А
х
&coarse/coarse/conv4-bn/AssignMovingAvg	AssignSubcoarse/conv4-bn/moving_mean*coarse/coarse/conv4-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:А
┤
-coarse/coarse/conv4-bn/AssignMovingAvg_1/readIdentitycoarse/conv4-bn/moving_variance*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
с
,coarse/coarse/conv4-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv4-bn/AssignMovingAvg_1/read#coarse/coarse/conv4-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
█
,coarse/coarse/conv4-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv4-bn/AssignMovingAvg_1/Subcoarse/coarse/conv4-bn/Squeeze*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
T0*
_output_shapes	
:А
ё
(coarse/coarse/conv4-bn/AssignMovingAvg_1	AssignSubcoarse/conv4-bn/moving_variance,coarse/coarse/conv4-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:А
~
coarse/coarse/conv4-reluRelu!coarse/coarse/conv4-bn/cond/Merge*
T0*0
_output_shapes
:         А
╔
coarse/coarse/MaxPool_2MaxPoolcoarse/coarse/conv4-relu*0
_output_shapes
:         	А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
═
Acoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/shapeConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*%
valueB"	   	   А      *
_output_shapes
:
╕
@coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/meanConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
valueB
 *    *
_output_shapes
: 
║
Bcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/stddevConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
valueB
 *  А?*
_output_shapes
: 
н
Kcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormalAcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/shape*(
_output_shapes
:		АА*
dtype0*
seed2 *

seed *
T0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w
╜
?coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mulMulKcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/TruncatedNormalBcoarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/stddev*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
л
;coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normalAdd?coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mul@coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal/mean*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
┘
coarse/conv5-conv/conv5-conv-w
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
Ы
%coarse/conv5-conv/conv5-conv-w/AssignAssigncoarse/conv5-conv/conv5-conv-w;coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
╡
#coarse/conv5-conv/conv5-conv-w/readIdentitycoarse/conv5-conv/conv5-conv-w*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
▓
0coarse/conv5-conv/conv5-conv-b/Initializer/ConstConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
┐
coarse/conv5-conv/conv5-conv-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Г
%coarse/conv5-conv/conv5-conv-b/AssignAssigncoarse/conv5-conv/conv5-conv-b0coarse/conv5-conv/conv5-conv-b/Initializer/Const*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
и
#coarse/conv5-conv/conv5-conv-b/readIdentitycoarse/conv5-conv/conv5-conv-b*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
Н
3coarse/coarse/conv5-conv/conv5-conv/conv5-conv-convConv2Dcoarse/coarse/MaxPool_2#coarse/conv5-conv/conv5-conv-w/read*0
_output_shapes
:         	А*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
ю
7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_addBiasAdd3coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv#coarse/conv5-conv/conv5-conv-b/read*
data_formatNHWC*
T0*0
_output_shapes
:         	А
Я
&coarse/conv5-bn/gamma/Initializer/onesConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*  А?*
_output_shapes	
:А
н
coarse/conv5-bn/gamma
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
▐
coarse/conv5-bn/gamma/AssignAssigncoarse/conv5-bn/gamma&coarse/conv5-bn/gamma/Initializer/ones*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Н
coarse/conv5-bn/gamma/readIdentitycoarse/conv5-bn/gamma*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
Ю
&coarse/conv5-bn/beta/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
л
coarse/conv5-bn/beta
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
█
coarse/conv5-bn/beta/AssignAssigncoarse/conv5-bn/beta&coarse/conv5-bn/beta/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
К
coarse/conv5-bn/beta/readIdentitycoarse/conv5-bn/beta*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
м
-coarse/conv5-bn/moving_mean/Initializer/zerosConst*
dtype0*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
valueBА*    *
_output_shapes	
:А
╣
coarse/conv5-bn/moving_mean
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
shared_name 
ў
"coarse/conv5-bn/moving_mean/AssignAssigncoarse/conv5-bn/moving_mean-coarse/conv5-bn/moving_mean/Initializer/zeros*
validate_shape(*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Я
 coarse/conv5-bn/moving_mean/readIdentitycoarse/conv5-bn/moving_mean*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
│
0coarse/conv5-bn/moving_variance/Initializer/onesConst*
dtype0*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
valueBА*  А?*
_output_shapes	
:А
┴
coarse/conv5-bn/moving_variance
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
shared_name 
Ж
&coarse/conv5-bn/moving_variance/AssignAssigncoarse/conv5-bn/moving_variance0coarse/conv5-bn/moving_variance/Initializer/ones*
validate_shape(*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
л
$coarse/conv5-bn/moving_variance/readIdentitycoarse/conv5-bn/moving_variance*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
m
"coarse/coarse/conv5-bn/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
y
$coarse/coarse/conv5-bn/cond/switch_tIdentity$coarse/coarse/conv5-bn/cond/Switch:1*
T0
*
_output_shapes
:
w
$coarse/coarse/conv5-bn/cond/switch_fIdentity"coarse/coarse/conv5-bn/cond/Switch*
T0
*
_output_shapes
:
_
#coarse/coarse/conv5-bn/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:
Л
!coarse/coarse/conv5-bn/cond/ConstConst%^coarse/coarse/conv5-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
Н
#coarse/coarse/conv5-bn/cond/Const_1Const%^coarse/coarse/conv5-bn/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
╝
1coarse/coarse/conv5-bn/cond/FusedBatchNorm/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add*
T0*L
_output_shapes:
8:         	А:         	А
╒
3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id*(
_class
loc:@coarse/conv5-bn/gamma*
T0*"
_output_shapes
:А:А
╙
3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id*'
_class
loc:@coarse/conv5-bn/beta*
T0*"
_output_shapes
:А:А
║
*coarse/coarse/conv5-bn/cond/FusedBatchNormFusedBatchNorm3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2:1!coarse/coarse/conv5-bn/cond/Const#coarse/coarse/conv5-bn/cond/Const_1*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*L
_output_shapes:
8:         	А:А:А:А:А
╛
3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id*J
_class@
><loc:@coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add*
T0*L
_output_shapes:
8:         	А:         	А
╫
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id*(
_class
loc:@coarse/conv5-bn/gamma*
T0*"
_output_shapes
:А:А
╒
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id*'
_class
loc:@coarse/conv5-bn/beta*
T0*"
_output_shapes
:А:А
у
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_3Switch coarse/conv5-bn/moving_mean/read#coarse/coarse/conv5-bn/cond/pred_id*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*"
_output_shapes
:А:А
ы
5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4Switch$coarse/conv5-bn/moving_variance/read#coarse/coarse/conv5-bn/cond/pred_id*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*"
_output_shapes
:А:А
т
,coarse/coarse/conv5-bn/cond/FusedBatchNorm_1FusedBatchNorm3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_25coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:         	А:А:А:А:А
╩
!coarse/coarse/conv5-bn/cond/MergeMerge,coarse/coarse/conv5-bn/cond/FusedBatchNorm_1*coarse/coarse/conv5-bn/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:         	А: 
╗
#coarse/coarse/conv5-bn/cond/Merge_1Merge.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:1,coarse/coarse/conv5-bn/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:А: 
╗
#coarse/coarse/conv5-bn/cond/Merge_2Merge.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:2,coarse/coarse/conv5-bn/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes
	:А: 
l
'coarse/coarse/conv5-bn/ExpandDims/inputConst*
dtype0*
valueB
 *
╫#<*
_output_shapes
: 
g
%coarse/coarse/conv5-bn/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
!coarse/coarse/conv5-bn/ExpandDims
ExpandDims'coarse/coarse/conv5-bn/ExpandDims/input%coarse/coarse/conv5-bn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
n
)coarse/coarse/conv5-bn/ExpandDims_1/inputConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
'coarse/coarse/conv5-bn/ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 
╢
#coarse/coarse/conv5-bn/ExpandDims_1
ExpandDims)coarse/coarse/conv5-bn/ExpandDims_1/input'coarse/coarse/conv5-bn/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
n
$coarse/coarse/conv5-bn/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
П
coarse/coarse/conv5-bn/ReshapeReshapeis_training$coarse/coarse/conv5-bn/Reshape/shape*
Tshape0*
T0
*
_output_shapes
:
┤
coarse/coarse/conv5-bn/SelectSelectcoarse/coarse/conv5-bn/Reshape!coarse/coarse/conv5-bn/ExpandDims#coarse/coarse/conv5-bn/ExpandDims_1*
T0*
_output_shapes
:
А
coarse/coarse/conv5-bn/SqueezeSqueezecoarse/coarse/conv5-bn/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
к
+coarse/coarse/conv5-bn/AssignMovingAvg/readIdentitycoarse/conv5-bn/moving_mean*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
┘
*coarse/coarse/conv5-bn/AssignMovingAvg/SubSub+coarse/coarse/conv5-bn/AssignMovingAvg/read#coarse/coarse/conv5-bn/cond/Merge_1*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
╙
*coarse/coarse/conv5-bn/AssignMovingAvg/MulMul*coarse/coarse/conv5-bn/AssignMovingAvg/Subcoarse/coarse/conv5-bn/Squeeze*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
T0*
_output_shapes	
:А
х
&coarse/coarse/conv5-bn/AssignMovingAvg	AssignSubcoarse/conv5-bn/moving_mean*coarse/coarse/conv5-bn/AssignMovingAvg/Mul*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking( *
T0*
_output_shapes	
:А
┤
-coarse/coarse/conv5-bn/AssignMovingAvg_1/readIdentitycoarse/conv5-bn/moving_variance*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
с
,coarse/coarse/conv5-bn/AssignMovingAvg_1/SubSub-coarse/coarse/conv5-bn/AssignMovingAvg_1/read#coarse/coarse/conv5-bn/cond/Merge_2*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
█
,coarse/coarse/conv5-bn/AssignMovingAvg_1/MulMul,coarse/coarse/conv5-bn/AssignMovingAvg_1/Subcoarse/coarse/conv5-bn/Squeeze*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
T0*
_output_shapes	
:А
ё
(coarse/coarse/conv5-bn/AssignMovingAvg_1	AssignSubcoarse/conv5-bn/moving_variance,coarse/coarse/conv5-bn/AssignMovingAvg_1/Mul*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking( *
T0*
_output_shapes	
:А
~
coarse/coarse/conv5-reluRelu!coarse/coarse/conv5-bn/cond/Merge*
T0*0
_output_shapes
:         	А
╔
coarse/coarse/MaxPool_3MaxPoolcoarse/coarse/conv5-relu*0
_output_shapes
:         А*
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
valueB"     ╠ *
_output_shapes
:
Ш
coarse/coarse/ReshapeReshapecoarse/coarse/MaxPool_3coarse/coarse/Reshape/shape*
Tshape0*
T0*)
_output_shapes
:         АШ
й
3coarse/fc1/fc1-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB" ╠    *
_output_shapes
:
Ь
2coarse/fc1/fc1-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *    *
_output_shapes
: 
Ю
4coarse/fc1/fc1-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w*
valueB
 *  А?*
_output_shapes
: 
№
=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc1/fc1-w/Initializer/truncated_normal/shape*!
_output_shapes
:АША*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc1/fc1-w
■
1coarse/fc1/fc1-w/Initializer/truncated_normal/mulMul=coarse/fc1/fc1-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc1/fc1-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
ь
-coarse/fc1/fc1-w/Initializer/truncated_normalAdd1coarse/fc1/fc1-w/Initializer/truncated_normal/mul2coarse/fc1/fc1-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
п
coarse/fc1/fc1-w
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
▄
coarse/fc1/fc1-w/AssignAssigncoarse/fc1/fc1-w-coarse/fc1/fc1-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
Д
coarse/fc1/fc1-w/readIdentitycoarse/fc1/fc1-w*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
Ц
"coarse/fc1/fc1-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
г
coarse/fc1/fc1-b
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
╦
coarse/fc1/fc1-b/AssignAssigncoarse/fc1/fc1-b"coarse/fc1/fc1-b/Initializer/Const*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
~
coarse/fc1/fc1-b/readIdentitycoarse/fc1/fc1-b*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
о
coarse/coarse/fc1/fc1/fc1-matMatMulcoarse/coarse/Reshapecoarse/fc1/fc1-w/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
н
"coarse/coarse/fc1/fc1/fc1-bias_addBiasAddcoarse/coarse/fc1/fc1/fc1-matcoarse/fc1/fc1-b/read*
data_formatNHWC*
T0*(
_output_shapes
:         А
й
3coarse/fc2/fc2-w/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB"      *
_output_shapes
:
Ь
2coarse/fc2/fc2-w/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *    *
_output_shapes
: 
Ю
4coarse/fc2/fc2-w/Initializer/truncated_normal/stddevConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB
 *  А?*
_output_shapes
: 
·
=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3coarse/fc2/fc2-w/Initializer/truncated_normal/shape*
_output_shapes
:	А*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@coarse/fc2/fc2-w
№
1coarse/fc2/fc2-w/Initializer/truncated_normal/mulMul=coarse/fc2/fc2-w/Initializer/truncated_normal/TruncatedNormal4coarse/fc2/fc2-w/Initializer/truncated_normal/stddev*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
ъ
-coarse/fc2/fc2-w/Initializer/truncated_normalAdd1coarse/fc2/fc2-w/Initializer/truncated_normal/mul2coarse/fc2/fc2-w/Initializer/truncated_normal/mean*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
л
coarse/fc2/fc2-w
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
┌
coarse/fc2/fc2-w/AssignAssigncoarse/fc2/fc2-w-coarse/fc2/fc2-w/Initializer/truncated_normal*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
В
coarse/fc2/fc2-w/readIdentitycoarse/fc2/fc2-w*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
Ф
"coarse/fc2/fc2-b/Initializer/ConstConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
б
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
╩
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
║
coarse/coarse/fc2/fc2/fc2-matMatMul"coarse/coarse/fc1/fc1/fc1-bias_addcoarse/fc2/fc2-w/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
м
"coarse/coarse/fc2/fc2/fc2-bias_addBiasAddcoarse/coarse/fc2/fc2/fc2-matcoarse/fc2/fc2-b/read*
data_formatNHWC*
T0*'
_output_shapes
:         
g
subSub"coarse/coarse/fc2/fc2/fc2-bias_addlabel*
T0*'
_output_shapes
:         
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
:         
J
add/yConst*
dtype0*
valueB
 *╠╝М+*
_output_shapes
: 
H
addAddPowadd/y*
T0*'
_output_shapes
:         
C
SqrtSqrtadd*
T0*'
_output_shapes
:         
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
в
gradients/ShapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
д
gradients/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
┬
!gradients/Mean_grad/Reshape/shapeConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB"      *
_output_shapes
:
Р
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
н
gradients/Mean_grad/ShapeShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
п
gradients/Mean_grad/Shape_1ShapeSqrt'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
о
gradients/Mean_grad/Shape_2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
у
gradients/Mean_grad/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
╞
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
х
gradients/Mean_grad/Const_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
╩
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
▀
gradients/Mean_grad/Maximum/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
▓
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
░
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
}
gradients/Sqrt_grad/SqrtGradSqrtGradSqrtgradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
л
gradients/add_grad/ShapeShapePow'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
н
gradients/add_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
е
gradients/add_grad/SumSumgradients/Sqrt_grad/SqrtGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
й
gradients/add_grad/Sum_1Sumgradients/Sqrt_grad/SqrtGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
М
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
╖
#gradients/add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:         
╧
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
л
gradients/Pow_grad/ShapeShapesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
н
gradients/Pow_grad/Shape_1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB *
_output_shapes
: 
┤
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
gradients/Pow_grad/mulMul+gradients/add_grad/tuple/control_dependencyPow/y*
T0*'
_output_shapes
:         
н
gradients/Pow_grad/sub/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *  А?*
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
:         
Б
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:         
б
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ч
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
▒
gradients/Pow_grad/Greater/yConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:         
д
gradients/Pow_grad/LogLogsub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:         
▒
gradients/Pow_grad/zeros_like	ZerosLikesub'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*'
_output_shapes
:         
и
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:         
Г
gradients/Pow_grad/mul_2Mul+gradients/add_grad/tuple/control_dependencyPow*
T0*'
_output_shapes
:         
Ж
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:         
е
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
М
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
╖
#gradients/Pow_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
┌
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*-
_class#
!loc:@gradients/Pow_grad/Reshape*
T0*'
_output_shapes
:         
╧
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
T0*
_output_shapes
: 
╩
gradients/sub_grad/ShapeShape"coarse/coarse/fc2/fc2/fc2-bias_add'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
п
gradients/sub_grad/Shape_1Shapelabel'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ч
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:         
╕
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
Ы
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:         
╖
#gradients/sub_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
┌
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:         
р
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:         
╡
=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradBiasAddGrad+gradients/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:
И
Bgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1,^gradients/sub_grad/tuple/control_dependency>^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad
й
Jgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencyIdentity+gradients/sub_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:         
╙
Lgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/BiasAddGrad*
T0*
_output_shapes
:
∙
3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependencycoarse/fc2/fc2-w/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         А
 
5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1MatMul"coarse/coarse/fc1/fc1/fc1-bias_addJgradients/coarse/coarse/fc2/fc2/fc2-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	А
Г
=gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul6^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1
┴
Egradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:         А
╛
Ggradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1>^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul_1*
T0*
_output_shapes
:	А
╨
=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradBiasAddGradEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes	
:А
в
Bgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1F^gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency>^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad
▌
Jgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencyIdentityEgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependencyC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc2/fc2/fc2-mat_grad/MatMul*
T0*(
_output_shapes
:         А
╘
Lgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1Identity=gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGradC^gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/group_deps*P
_classF
DBloc:@gradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
·
3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMulMatMulJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependencycoarse/fc1/fc1-w/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:         АШ
Ї
5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1MatMulcoarse/coarse/ReshapeJgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:АША
Г
=gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_14^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul6^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1
┬
Egradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependencyIdentity3gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*F
_class<
:8loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul*
T0*)
_output_shapes
:         АШ
└
Ggradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1Identity5gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1>^gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/group_deps*H
_class>
<:loc:@gradients/coarse/coarse/fc1/fc1/fc1-mat_grad/MatMul_1*
T0*!
_output_shapes
:АША
╤
*gradients/coarse/coarse/Reshape_grad/ShapeShapecoarse/coarse/MaxPool_3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
out_type0*
T0*
_output_shapes
:
є
,gradients/coarse/coarse/Reshape_grad/ReshapeReshapeEgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency*gradients/coarse/coarse/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:         А
п
2gradients/coarse/coarse/MaxPool_3_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv5-relucoarse/coarse/MaxPool_3,gradients/coarse/coarse/Reshape_grad/Reshape*0
_output_shapes
:         	А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┼
0gradients/coarse/coarse/conv5-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_3_grad/MaxPoolGradcoarse/coarse/conv5-relu*
T0*0
_output_shapes
:         	А
╖
:gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv5-relu_grad/ReluGrad#coarse/coarse/conv5-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:         	А:         	А
╓
Agradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_grad
╒
Igradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*0
_output_shapes
:         	А
┘
Kgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv5-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv5-relu_grad/ReluGrad*
T0*0
_output_shapes
:         	А
╟
gradients/zeros_like	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_1	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_2	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_3	ZerosLike.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Ь
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:         	А:А:А:А:А
ї
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Э
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         	А
М
Vgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
М
Vgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_4	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_5	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_6	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╟
gradients/zeros_like_7	ZerosLike,coarse/coarse/conv5-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Д
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv5-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv5-bn/cond/FusedBatchNorm:3,coarse/coarse/conv5-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*F
_output_shapes4
2:         	А:А:А: : 
ё
Jgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Х
Rgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         	А
Д
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Д
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Б
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Я
gradients/SwitchSwitch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         	А:         	А
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
T0*
_output_shapes
:
к
gradients/zeros/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*0
_output_shapes
:         	А
В
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*
T0*2
_output_shapes 
:         	А: 
┌
gradients/Switch_1Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_2Shapegradients/Switch_1:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_1/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
N*
T0*
_output_shapes
	:А: 
┘
gradients/Switch_2Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_2/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
N*
T0*
_output_shapes
	:А: 
б
gradients/Switch_3Switch7coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         	А:         	А
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_3/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
А
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*0
_output_shapes
:         	А
А
Jgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*2
_output_shapes 
:         	А: 
┌
gradients/Switch_4Switchcoarse/conv5-bn/gamma/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_4/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes	
:А
я
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
N*
T0*
_output_shapes
	:А: 
┘
gradients/Switch_5Switchcoarse/conv5-bn/beta/read#coarse/coarse/conv5-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_5/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes	
:А
я
Lgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
N*
T0*
_output_shapes
	:А: 
╒
gradients/AddNAddNLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         	А
о
Rgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
T0*
_output_shapes	
:А
Х
Wgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddNS^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGrad
ё
_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddNX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         	А
и
agradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
╚
gradients/AddN_1AddNNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:А
╚
gradients/AddN_2AddNNgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:А
е
Igradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_2#coarse/conv5-conv/conv5-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ъ
Vgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeN#coarse/conv5-conv/conv5-conv-w/read_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
т
Wgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_2Kgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilter
╗
[gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropInput*
T0*0
_output_shapes
:         	А
╖
]gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/Conv2DBackpropFilter*
T0*(
_output_shapes
:		АА
▐
2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv4-relucoarse/coarse/MaxPool_2[gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency*0
_output_shapes
:         А*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┼
0gradients/coarse/coarse/conv4-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_2_grad/MaxPoolGradcoarse/coarse/conv4-relu*
T0*0
_output_shapes
:         А
╖
:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv4-relu_grad/ReluGrad#coarse/coarse/conv4-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*L
_output_shapes:
8:         А:         А
╓
Agradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad
╒
Igradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:         А
┘
Kgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv4-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv4-relu_grad/ReluGrad*
T0*0
_output_shapes
:         А
╔
gradients/zeros_like_8	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╔
gradients/zeros_like_9	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╩
gradients/zeros_like_10	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╩
gradients/zeros_like_11	ZerosLike.coarse/coarse/conv4-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Ь
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *L
_output_shapes:
8:         А:А:А:А:А
ї
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Э
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         А
М
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
М
Vgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_12	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_13	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_14	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
╚
gradients/zeros_like_15	ZerosLike,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes	
:А
Д
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv4-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv4-bn/cond/FusedBatchNorm:3,coarse/coarse/conv4-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*F
_output_shapes4
2:         А:А:А: : 
ё
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Х
Rgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*0
_output_shapes
:         А
Д
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Д
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes	
:А
Б
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
б
gradients/Switch_6Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         А:         А
e
gradients/Shape_7Shapegradients/Switch_6:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_6/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
А
gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*0
_output_shapes
:         А
Д
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*
T0*2
_output_shapes 
:         А: 
┌
gradients/Switch_7Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_8Shapegradients/Switch_7:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_7/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
N*
T0*
_output_shapes
	:А: 
┘
gradients/Switch_8Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_9Shapegradients/Switch_8:1*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_8/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
k
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes	
:А
є
Ngradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
N*
T0*
_output_shapes
	:А: 
б
gradients/Switch_9Switch7coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*L
_output_shapes:
8:         А:         А
d
gradients/Shape_10Shapegradients/Switch_9*
out_type0*
T0*
_output_shapes
:
м
gradients/zeros_9/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
Б
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*0
_output_shapes
:         А
А
Jgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
N*
T0*2
_output_shapes 
:         А: 
█
gradients/Switch_10Switchcoarse/conv4-bn/gamma/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_11Shapegradients/Switch_10*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_10/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
n
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes	
:А
Ё
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
N*
T0*
_output_shapes
	:А: 
┌
gradients/Switch_11Switchcoarse/conv4-bn/beta/read#coarse/coarse/conv4-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*"
_output_shapes
:А:А
e
gradients/Shape_12Shapegradients/Switch_11*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_11/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
n
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*
_output_shapes	
:А
Ё
Lgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
N*
T0*
_output_shapes
	:А: 
╫
gradients/AddN_3AddNLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         А
░
Rgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
T0*
_output_shapes	
:А
Ч
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_3S^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad
є
_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_3X^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*0
_output_shapes
:         А
и
agradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes	
:А
╚
gradients/AddN_4AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes	
:А
╚
gradients/AddN_5AddNNgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes	
:А
е
Igradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool_1#coarse/conv4-conv/conv4-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ъ
Vgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN#coarse/conv4-conv/conv4-conv-w/read_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
т
Wgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPool_1Kgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:         @
╢
]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@А
▌
2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv3-relucoarse/coarse/MaxPool_1[gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:          $@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
─
0gradients/coarse/coarse/conv3-relu_grad/ReluGradReluGrad2gradients/coarse/coarse/MaxPool_1_grad/MaxPoolGradcoarse/coarse/conv3-relu*
T0*/
_output_shapes
:          $@
╡
:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv3-relu_grad/ReluGrad#coarse/coarse/conv3-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:          $@:          $@
╓
Agradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad
╘
Igradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:          $@
╪
Kgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv3-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv3-relu_grad/ReluGrad*
T0*/
_output_shapes
:          $@
╔
gradients/zeros_like_16	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_17	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_18	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╔
gradients/zeros_like_19	ZerosLike.coarse/coarse/conv3-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
Ч
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:          $@:@:@:@:@
ї
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Ь
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:          $@
Л
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Л
Vgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
╟
gradients/zeros_like_20	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_21	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_22	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
╟
gradients/zeros_like_23	ZerosLike,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
:@
Б
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv3-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv3-bn/cond/FusedBatchNorm:3,coarse/coarse/conv3-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*C
_output_shapes1
/:          $@:@:@: : 
ё
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Ф
Rgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:          $@
Г
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Г
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
:@
Б
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
а
gradients/Switch_12Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:          $@:          $@
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_12/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*/
_output_shapes
:          $@
Д
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
N*
T0*1
_output_shapes
:          $@: 
┘
gradients/Switch_13Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
g
gradients/Shape_14Shapegradients/Switch_13:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_13/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
:@
є
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
N*
T0*
_output_shapes

:@: 
╪
gradients/Switch_14Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_14/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
:@
є
Ngradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
N*
T0*
_output_shapes

:@: 
а
gradients/Switch_15Switch7coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:          $@:          $@
e
gradients/Shape_16Shapegradients/Switch_15*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_15/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*/
_output_shapes
:          $@
А
Jgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
N*
T0*1
_output_shapes
:          $@: 
┘
gradients/Switch_16Switchcoarse/conv3-bn/gamma/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_17Shapegradients/Switch_16*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_16/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
:@
я
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
N*
T0*
_output_shapes

:@: 
╪
gradients/Switch_17Switchcoarse/conv3-bn/beta/read#coarse/coarse/conv3-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
:@:@
e
gradients/Shape_18Shapegradients/Switch_17*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_17/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
:@
я
Lgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
N*
T0*
_output_shapes

:@: 
╓
gradients/AddN_6AddNLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:          $@
п
Rgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_6*
data_formatNHWC*
T0*
_output_shapes
:@
Ч
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_6S^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad
Є
_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_6X^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:          $@
з
agradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:@
╟
gradients/AddN_7AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
:@
╟
gradients/AddN_8AddNNgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv3-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
:@
г
Igradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeNShapeNcoarse/coarse/MaxPool#coarse/conv3-conv/conv3-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ъ
Vgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN#coarse/conv3-conv/conv3-conv-w/read_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
р
Wgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/MaxPoolKgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:          $ 
╡
]gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
┘
0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradMaxPoolGradcoarse/coarse/conv2-relucoarse/coarse/MaxPool[gradients/coarse/coarse/conv3-conv/conv3-conv/conv3-conv-conv_grad/tuple/control_dependency*/
_output_shapes
:         @H *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
┬
0gradients/coarse/coarse/conv2-relu_grad/ReluGradReluGrad0gradients/coarse/coarse/MaxPool_grad/MaxPoolGradcoarse/coarse/conv2-relu*
T0*/
_output_shapes
:         @H 
╡
:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradSwitch0gradients/coarse/coarse/conv2-relu_grad/ReluGrad#coarse/coarse/conv2-bn/cond/pred_id*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*J
_output_shapes8
6:         @H :         @H 
╓
Agradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1;^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad
╘
Igradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_gradB^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*/
_output_shapes
:         @H 
╪
Kgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/coarse/coarse/conv2-bn/cond/Merge_grad/cond_grad:1B^gradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/group_deps*C
_class9
75loc:@gradients/coarse/coarse/conv2-relu_grad/ReluGrad*
T0*/
_output_shapes
:         @H 
╔
gradients/zeros_like_24	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_25	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_26	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╔
gradients/zeros_like_27	ZerosLike.coarse/coarse/conv2-bn/cond/FusedBatchNorm_1:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
Ч
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradIgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency3coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch5coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_15coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_35coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training( *G
_output_shapes5
3:         @H : : : : 
ї
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1O^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
Ь
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGradM^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:         @H 
Л
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Л
Vgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityPgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/group_deps*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
╟
gradients/zeros_like_28	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:1'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_29	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:2'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_30	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
╟
gradients/zeros_like_31	ZerosLike,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*
_output_shapes
: 
Б
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradKgradients/coarse/coarse/conv2-bn/cond/Merge_grad/tuple/control_dependency_13coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch:15coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1:1,coarse/coarse/conv2-bn/cond/FusedBatchNorm:3,coarse/coarse/conv2-bn/cond/FusedBatchNorm:4*
epsilon%oГ:*
data_formatNHWC*
T0*
is_training(*C
_output_shapes1
/:         @H : : : : 
ё
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1M^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad
Ф
Rgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGradK^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*/
_output_shapes
:         @H 
Г
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Г
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
Б
Tgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4K^gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
T0*
_output_shapes
: 
а
gradients/Switch_18Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:         @H :         @H 
g
gradients/Shape_19Shapegradients/Switch_18:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_18/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*
T0*/
_output_shapes
:         @H 
Д
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*
N*
T0*1
_output_shapes
:         @H : 
┘
gradients/Switch_19Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_20Shapegradients/Switch_19:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_19/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
T0*
_output_shapes
: 
є
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
N*
T0*
_output_shapes

: : 
╪
gradients/Switch_20Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
g
gradients/Shape_21Shapegradients/Switch_20:1*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_20/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*
_output_shapes
: 
є
Ngradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeVgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
N*
T0*
_output_shapes

: : 
а
gradients/Switch_21Switch7coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0*J
_output_shapes8
6:         @H :         @H 
e
gradients/Shape_22Shapegradients/Switch_21*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_21/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
В
gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*
T0*/
_output_shapes
:         @H 
А
Jgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_gradMergeRgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_21*
N*
T0*1
_output_shapes
:         @H : 
┘
gradients/Switch_22Switchcoarse/conv2-bn/gamma/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_23Shapegradients/Switch_22*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_22/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*
T0*
_output_shapes
: 
я
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_22*
N*
T0*
_output_shapes

: : 
╪
gradients/Switch_23Switchcoarse/conv2-bn/beta/read#coarse/coarse/conv2-bn/cond/pred_id'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
T0* 
_output_shapes
: : 
e
gradients/Shape_24Shapegradients/Switch_23*
out_type0*
T0*
_output_shapes
:
н
gradients/zeros_23/ConstConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *    *
_output_shapes
: 
m
gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
T0*
_output_shapes
: 
я
Lgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeTgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_23*
N*
T0*
_output_shapes

: : 
╓
gradients/AddN_9AddNLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_gradJgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:         @H 
п
Rgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradBiasAddGradgradients/AddN_9*
data_formatNHWC*
T0*
_output_shapes
: 
Ч
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1^gradients/AddN_9S^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad
Є
_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependencyIdentitygradients/AddN_9X^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*_
_classU
SQloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
T0*/
_output_shapes
:         @H 
з
agradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency_1IdentityRgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGradX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/group_deps*e
_class[
YWloc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
: 
╚
gradients/AddN_10AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
T0*
_output_shapes
: 
╚
gradients/AddN_11AddNNgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradLgradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*a
_classW
USloc:@gradients/coarse/coarse/conv2-bn/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
T0*
_output_shapes
: 
б
Igradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeNShapeNcoarse/coarse/relu1#coarse/conv2-conv/conv2-conv-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
ъ
Vgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputConv2DBackpropInputIgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN#coarse/conv2-conv/conv2-conv-w/read_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Wgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterConv2DBackpropFiltercoarse/coarse/relu1Kgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/ShapeN:1_gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▐
Sgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1W^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputX^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter
║
[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencyIdentityVgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInputT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*i
_class_
][loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:         @H
╡
]gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependency_1IdentityWgradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilterT^gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/group_deps*j
_class`
^\loc:@gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
у
+gradients/coarse/coarse/relu1_grad/ReluGradReluGrad[gradients/coarse/coarse/conv2-conv/conv2-conv/conv2-conv-conv_grad/tuple/control_dependencycoarse/coarse/relu1*
T0*/
_output_shapes
:         @H
╗
Cgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradBiasAddGrad+gradients/coarse/coarse/relu1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
Ф
Hgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1,^gradients/coarse/coarse/relu1_grad/ReluGradD^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad
╬
Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependencyIdentity+gradients/coarse/coarse/relu1_grad/ReluGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*>
_class4
20loc:@gradients/coarse/coarse/relu1_grad/ReluGrad*
T0*/
_output_shapes
:         @H
ы
Rgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency_1IdentityCgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGradI^gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/group_deps*V
_classL
JHloc:@gradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/BiasAddGrad*
T0*
_output_shapes
:
°
:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNShapeNimgcoarse/conv1/conv1-w/read'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
N*
out_type0*
T0* 
_output_shapes
::
│
Ggradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputConv2DBackpropInput:gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeNcoarse/conv1/conv1-w/readPgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
б
Hgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterConv2DBackpropFilterimg<gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/ShapeN:1Pgradients/coarse/coarse/conv1/conv1/conv1-biad_add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
▒
Dgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_depsNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1H^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputI^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter
А
Lgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependencyIdentityGgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInputE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropInput*
T0*1
_output_shapes
:         АР
∙
Ngradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1IdentityHgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilterE^gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/coarse/coarse/conv1/conv1/conv1-conv_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
З
beta1_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *fff?*
_output_shapes
: 
Ш
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
╖
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
З
beta2_power/initial_valueConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB
 *w╛?*
_output_shapes
: 
Ш
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
╖
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
╣
+coarse/conv1/conv1-w/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
╞
coarse/conv1/conv1-w/Adam
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
ї
 coarse/conv1/conv1-w/Adam/AssignAssigncoarse/conv1/conv1-w/Adam+coarse/conv1/conv1-w/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Я
coarse/conv1/conv1-w/Adam/readIdentitycoarse/conv1/conv1-w/Adam*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
╗
-coarse/conv1/conv1-w/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-w*%
valueB*    *&
_output_shapes
:
╚
coarse/conv1/conv1-w/Adam_1
VariableV2*
	container *&
_output_shapes
:*
dtype0*
shape:*'
_class
loc:@coarse/conv1/conv1-w*
shared_name 
√
"coarse/conv1/conv1-w/Adam_1/AssignAssigncoarse/conv1/conv1-w/Adam_1-coarse/conv1/conv1-w/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
г
 coarse/conv1/conv1-w/Adam_1/readIdentitycoarse/conv1/conv1-w/Adam_1*'
_class
loc:@coarse/conv1/conv1-w*
T0*&
_output_shapes
:
б
+coarse/conv1/conv1-b/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
о
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
щ
 coarse/conv1/conv1-b/Adam/AssignAssigncoarse/conv1/conv1-b/Adam+coarse/conv1/conv1-b/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
У
coarse/conv1/conv1-b/Adam/readIdentitycoarse/conv1/conv1-b/Adam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
г
-coarse/conv1/conv1-b/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv1/conv1-b*
valueB*    *
_output_shapes
:
░
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
я
"coarse/conv1/conv1-b/Adam_1/AssignAssigncoarse/conv1/conv1-b/Adam_1-coarse/conv1/conv1-b/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Ч
 coarse/conv1/conv1-b/Adam_1/readIdentitycoarse/conv1/conv1-b/Adam_1*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
:
═
5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
┌
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
Э
*coarse/conv2-conv/conv2-conv-w/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-w/Adam5coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
╜
(coarse/conv2-conv/conv2-conv-w/Adam/readIdentity#coarse/conv2-conv/conv2-conv-w/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╧
7coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*%
valueB *    *&
_output_shapes
: 
▄
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
г
,coarse/conv2-conv/conv2-conv-w/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-w/Adam_17coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
┴
*coarse/conv2-conv/conv2-conv-w/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
T0*&
_output_shapes
: 
╡
5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
┬
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
С
*coarse/conv2-conv/conv2-conv-b/Adam/AssignAssign#coarse/conv2-conv/conv2-conv-b/Adam5coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
▒
(coarse/conv2-conv/conv2-conv-b/Adam/readIdentity#coarse/conv2-conv/conv2-conv-b/Adam*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
╖
7coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
valueB *    *
_output_shapes
: 
─
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
Ч
,coarse/conv2-conv/conv2-conv-b/Adam_1/AssignAssign%coarse/conv2-conv/conv2-conv-b/Adam_17coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
╡
*coarse/conv2-conv/conv2-conv-b/Adam_1/readIdentity%coarse/conv2-conv/conv2-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
T0*
_output_shapes
: 
г
,coarse/conv2-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
░
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
э
!coarse/conv2-bn/gamma/Adam/AssignAssigncoarse/conv2-bn/gamma/Adam,coarse/conv2-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Ц
coarse/conv2-bn/gamma/Adam/readIdentitycoarse/conv2-bn/gamma/Adam*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
е
.coarse/conv2-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv2-bn/gamma*
valueB *    *
_output_shapes
: 
▓
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
є
#coarse/conv2-bn/gamma/Adam_1/AssignAssigncoarse/conv2-bn/gamma/Adam_1.coarse/conv2-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Ъ
!coarse/conv2-bn/gamma/Adam_1/readIdentitycoarse/conv2-bn/gamma/Adam_1*(
_class
loc:@coarse/conv2-bn/gamma*
T0*
_output_shapes
: 
б
+coarse/conv2-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
о
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
щ
 coarse/conv2-bn/beta/Adam/AssignAssigncoarse/conv2-bn/beta/Adam+coarse/conv2-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
У
coarse/conv2-bn/beta/Adam/readIdentitycoarse/conv2-bn/beta/Adam*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
г
-coarse/conv2-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv2-bn/beta*
valueB *    *
_output_shapes
: 
░
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
я
"coarse/conv2-bn/beta/Adam_1/AssignAssigncoarse/conv2-bn/beta/Adam_1-coarse/conv2-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
Ч
 coarse/conv2-bn/beta/Adam_1/readIdentitycoarse/conv2-bn/beta/Adam_1*'
_class
loc:@coarse/conv2-bn/beta*
T0*
_output_shapes
: 
═
5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
┌
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
Э
*coarse/conv3-conv/conv3-conv-w/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-w/Adam5coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
╜
(coarse/conv3-conv/conv3-conv-w/Adam/readIdentity#coarse/conv3-conv/conv3-conv-w/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╧
7coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*%
valueB @*    *&
_output_shapes
: @
▄
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
г
,coarse/conv3-conv/conv3-conv-w/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-w/Adam_17coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
┴
*coarse/conv3-conv/conv3-conv-w/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
T0*&
_output_shapes
: @
╡
5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
┬
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
С
*coarse/conv3-conv/conv3-conv-b/Adam/AssignAssign#coarse/conv3-conv/conv3-conv-b/Adam5coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
▒
(coarse/conv3-conv/conv3-conv-b/Adam/readIdentity#coarse/conv3-conv/conv3-conv-b/Adam*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
╖
7coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
valueB@*    *
_output_shapes
:@
─
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
Ч
,coarse/conv3-conv/conv3-conv-b/Adam_1/AssignAssign%coarse/conv3-conv/conv3-conv-b/Adam_17coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
╡
*coarse/conv3-conv/conv3-conv-b/Adam_1/readIdentity%coarse/conv3-conv/conv3-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
T0*
_output_shapes
:@
г
,coarse/conv3-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
░
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
э
!coarse/conv3-bn/gamma/Adam/AssignAssigncoarse/conv3-bn/gamma/Adam,coarse/conv3-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Ц
coarse/conv3-bn/gamma/Adam/readIdentitycoarse/conv3-bn/gamma/Adam*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
е
.coarse/conv3-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv3-bn/gamma*
valueB@*    *
_output_shapes
:@
▓
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
є
#coarse/conv3-bn/gamma/Adam_1/AssignAssigncoarse/conv3-bn/gamma/Adam_1.coarse/conv3-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Ъ
!coarse/conv3-bn/gamma/Adam_1/readIdentitycoarse/conv3-bn/gamma/Adam_1*(
_class
loc:@coarse/conv3-bn/gamma*
T0*
_output_shapes
:@
б
+coarse/conv3-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
о
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
щ
 coarse/conv3-bn/beta/Adam/AssignAssigncoarse/conv3-bn/beta/Adam+coarse/conv3-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
У
coarse/conv3-bn/beta/Adam/readIdentitycoarse/conv3-bn/beta/Adam*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
г
-coarse/conv3-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv3-bn/beta*
valueB@*    *
_output_shapes
:@
░
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
я
"coarse/conv3-bn/beta/Adam_1/AssignAssigncoarse/conv3-bn/beta/Adam_1-coarse/conv3-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
Ч
 coarse/conv3-bn/beta/Adam_1/readIdentitycoarse/conv3-bn/beta/Adam_1*'
_class
loc:@coarse/conv3-bn/beta*
T0*
_output_shapes
:@
╧
5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@А*    *'
_output_shapes
:@А
▄
#coarse/conv4-conv/conv4-conv-w/Adam
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
Ю
*coarse/conv4-conv/conv4-conv-w/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-w/Adam5coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
╛
(coarse/conv4-conv/conv4-conv-w/Adam/readIdentity#coarse/conv4-conv/conv4-conv-w/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╤
7coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*&
valueB@А*    *'
_output_shapes
:@А
▐
%coarse/conv4-conv/conv4-conv-w/Adam_1
VariableV2*
	container *'
_output_shapes
:@А*
dtype0*
shape:@А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
shared_name 
д
,coarse/conv4-conv/conv4-conv-w/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-w/Adam_17coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
┬
*coarse/conv4-conv/conv4-conv-w/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
T0*'
_output_shapes
:@А
╖
5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
─
#coarse/conv4-conv/conv4-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Т
*coarse/conv4-conv/conv4-conv-b/Adam/AssignAssign#coarse/conv4-conv/conv4-conv-b/Adam5coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
▓
(coarse/conv4-conv/conv4-conv-b/Adam/readIdentity#coarse/conv4-conv/conv4-conv-b/Adam*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
╣
7coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
valueBА*    *
_output_shapes	
:А
╞
%coarse/conv4-conv/conv4-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
shared_name 
Ш
,coarse/conv4-conv/conv4-conv-b/Adam_1/AssignAssign%coarse/conv4-conv/conv4-conv-b/Adam_17coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
╢
*coarse/conv4-conv/conv4-conv-b/Adam_1/readIdentity%coarse/conv4-conv/conv4-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
T0*
_output_shapes	
:А
е
,coarse/conv4-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv4-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
ю
!coarse/conv4-bn/gamma/Adam/AssignAssigncoarse/conv4-bn/gamma/Adam,coarse/conv4-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ч
coarse/conv4-bn/gamma/Adam/readIdentitycoarse/conv4-bn/gamma/Adam*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
з
.coarse/conv4-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv4-bn/gamma*
valueBА*    *
_output_shapes	
:А
┤
coarse/conv4-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv4-bn/gamma*
shared_name 
Ї
#coarse/conv4-bn/gamma/Adam_1/AssignAssigncoarse/conv4-bn/gamma/Adam_1.coarse/conv4-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ы
!coarse/conv4-bn/gamma/Adam_1/readIdentitycoarse/conv4-bn/gamma/Adam_1*(
_class
loc:@coarse/conv4-bn/gamma*
T0*
_output_shapes	
:А
г
+coarse/conv4-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
░
coarse/conv4-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
ъ
 coarse/conv4-bn/beta/Adam/AssignAssigncoarse/conv4-bn/beta/Adam+coarse/conv4-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ф
coarse/conv4-bn/beta/Adam/readIdentitycoarse/conv4-bn/beta/Adam*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
е
-coarse/conv4-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv4-bn/beta*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv4-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv4-bn/beta*
shared_name 
Ё
"coarse/conv4-bn/beta/Adam_1/AssignAssigncoarse/conv4-bn/beta/Adam_1-coarse/conv4-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ш
 coarse/conv4-bn/beta/Adam_1/readIdentitycoarse/conv4-bn/beta/Adam_1*'
_class
loc:@coarse/conv4-bn/beta*
T0*
_output_shapes	
:А
╤
5coarse/conv5-conv/conv5-conv-w/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*'
valueB		АА*    *(
_output_shapes
:		АА
▐
#coarse/conv5-conv/conv5-conv-w/Adam
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
Я
*coarse/conv5-conv/conv5-conv-w/Adam/AssignAssign#coarse/conv5-conv/conv5-conv-w/Adam5coarse/conv5-conv/conv5-conv-w/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
┐
(coarse/conv5-conv/conv5-conv-w/Adam/readIdentity#coarse/conv5-conv/conv5-conv-w/Adam*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
╙
7coarse/conv5-conv/conv5-conv-w/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*'
valueB		АА*    *(
_output_shapes
:		АА
р
%coarse/conv5-conv/conv5-conv-w/Adam_1
VariableV2*
	container *(
_output_shapes
:		АА*
dtype0*
shape:		АА*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
shared_name 
е
,coarse/conv5-conv/conv5-conv-w/Adam_1/AssignAssign%coarse/conv5-conv/conv5-conv-w/Adam_17coarse/conv5-conv/conv5-conv-w/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
├
*coarse/conv5-conv/conv5-conv-w/Adam_1/readIdentity%coarse/conv5-conv/conv5-conv-w/Adam_1*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
T0*(
_output_shapes
:		АА
╖
5coarse/conv5-conv/conv5-conv-b/Adam/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
─
#coarse/conv5-conv/conv5-conv-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Т
*coarse/conv5-conv/conv5-conv-b/Adam/AssignAssign#coarse/conv5-conv/conv5-conv-b/Adam5coarse/conv5-conv/conv5-conv-b/Adam/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
▓
(coarse/conv5-conv/conv5-conv-b/Adam/readIdentity#coarse/conv5-conv/conv5-conv-b/Adam*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
╣
7coarse/conv5-conv/conv5-conv-b/Adam_1/Initializer/zerosConst*
dtype0*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
valueBА*    *
_output_shapes	
:А
╞
%coarse/conv5-conv/conv5-conv-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
shared_name 
Ш
,coarse/conv5-conv/conv5-conv-b/Adam_1/AssignAssign%coarse/conv5-conv/conv5-conv-b/Adam_17coarse/conv5-conv/conv5-conv-b/Adam_1/Initializer/zeros*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
╢
*coarse/conv5-conv/conv5-conv-b/Adam_1/readIdentity%coarse/conv5-conv/conv5-conv-b/Adam_1*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
T0*
_output_shapes	
:А
е
,coarse/conv5-bn/gamma/Adam/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv5-bn/gamma/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
ю
!coarse/conv5-bn/gamma/Adam/AssignAssigncoarse/conv5-bn/gamma/Adam,coarse/conv5-bn/gamma/Adam/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ч
coarse/conv5-bn/gamma/Adam/readIdentitycoarse/conv5-bn/gamma/Adam*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
з
.coarse/conv5-bn/gamma/Adam_1/Initializer/zerosConst*
dtype0*(
_class
loc:@coarse/conv5-bn/gamma*
valueBА*    *
_output_shapes	
:А
┤
coarse/conv5-bn/gamma/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*(
_class
loc:@coarse/conv5-bn/gamma*
shared_name 
Ї
#coarse/conv5-bn/gamma/Adam_1/AssignAssigncoarse/conv5-bn/gamma/Adam_1.coarse/conv5-bn/gamma/Adam_1/Initializer/zeros*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Ы
!coarse/conv5-bn/gamma/Adam_1/readIdentitycoarse/conv5-bn/gamma/Adam_1*(
_class
loc:@coarse/conv5-bn/gamma*
T0*
_output_shapes	
:А
г
+coarse/conv5-bn/beta/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
░
coarse/conv5-bn/beta/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
ъ
 coarse/conv5-bn/beta/Adam/AssignAssigncoarse/conv5-bn/beta/Adam+coarse/conv5-bn/beta/Adam/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ф
coarse/conv5-bn/beta/Adam/readIdentitycoarse/conv5-bn/beta/Adam*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
е
-coarse/conv5-bn/beta/Adam_1/Initializer/zerosConst*
dtype0*'
_class
loc:@coarse/conv5-bn/beta*
valueBА*    *
_output_shapes	
:А
▓
coarse/conv5-bn/beta/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*'
_class
loc:@coarse/conv5-bn/beta*
shared_name 
Ё
"coarse/conv5-bn/beta/Adam_1/AssignAssigncoarse/conv5-bn/beta/Adam_1-coarse/conv5-bn/beta/Adam_1/Initializer/zeros*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
Ш
 coarse/conv5-bn/beta/Adam_1/readIdentitycoarse/conv5-bn/beta/Adam_1*'
_class
loc:@coarse/conv5-bn/beta*
T0*
_output_shapes	
:А
з
'coarse/fc1/fc1-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueBАША*    *!
_output_shapes
:АША
┤
coarse/fc1/fc1-w/Adam
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
р
coarse/fc1/fc1-w/Adam/AssignAssigncoarse/fc1/fc1-w/Adam'coarse/fc1/fc1-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
О
coarse/fc1/fc1-w/Adam/readIdentitycoarse/fc1/fc1-w/Adam*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
й
)coarse/fc1/fc1-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-w* 
valueBАША*    *!
_output_shapes
:АША
╢
coarse/fc1/fc1-w/Adam_1
VariableV2*
	container *!
_output_shapes
:АША*
dtype0*
shape:АША*#
_class
loc:@coarse/fc1/fc1-w*
shared_name 
ц
coarse/fc1/fc1-w/Adam_1/AssignAssigncoarse/fc1/fc1-w/Adam_1)coarse/fc1/fc1-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
Т
coarse/fc1/fc1-w/Adam_1/readIdentitycoarse/fc1/fc1-w/Adam_1*#
_class
loc:@coarse/fc1/fc1-w*
T0*!
_output_shapes
:АША
Ы
'coarse/fc1/fc1-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
и
coarse/fc1/fc1-b/Adam
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
┌
coarse/fc1/fc1-b/Adam/AssignAssigncoarse/fc1/fc1-b/Adam'coarse/fc1/fc1-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
И
coarse/fc1/fc1-b/Adam/readIdentitycoarse/fc1/fc1-b/Adam*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
Э
)coarse/fc1/fc1-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc1/fc1-b*
valueBА*    *
_output_shapes	
:А
к
coarse/fc1/fc1-b/Adam_1
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*#
_class
loc:@coarse/fc1/fc1-b*
shared_name 
р
coarse/fc1/fc1-b/Adam_1/AssignAssigncoarse/fc1/fc1-b/Adam_1)coarse/fc1/fc1-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
М
coarse/fc1/fc1-b/Adam_1/readIdentitycoarse/fc1/fc1-b/Adam_1*#
_class
loc:@coarse/fc1/fc1-b*
T0*
_output_shapes	
:А
г
'coarse/fc2/fc2-w/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	А*    *
_output_shapes
:	А
░
coarse/fc2/fc2-w/Adam
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
▐
coarse/fc2/fc2-w/Adam/AssignAssigncoarse/fc2/fc2-w/Adam'coarse/fc2/fc2-w/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
М
coarse/fc2/fc2-w/Adam/readIdentitycoarse/fc2/fc2-w/Adam*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
е
)coarse/fc2/fc2-w/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-w*
valueB	А*    *
_output_shapes
:	А
▓
coarse/fc2/fc2-w/Adam_1
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*#
_class
loc:@coarse/fc2/fc2-w*
shared_name 
ф
coarse/fc2/fc2-w/Adam_1/AssignAssigncoarse/fc2/fc2-w/Adam_1)coarse/fc2/fc2-w/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
Р
coarse/fc2/fc2-w/Adam_1/readIdentitycoarse/fc2/fc2-w/Adam_1*#
_class
loc:@coarse/fc2/fc2-w*
T0*
_output_shapes
:	А
Щ
'coarse/fc2/fc2-b/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
ж
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
┘
coarse/fc2/fc2-b/Adam/AssignAssigncoarse/fc2/fc2-b/Adam'coarse/fc2/fc2-b/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
З
coarse/fc2/fc2-b/Adam/readIdentitycoarse/fc2/fc2-b/Adam*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Ы
)coarse/fc2/fc2-b/Adam_1/Initializer/zerosConst*
dtype0*#
_class
loc:@coarse/fc2/fc2-b*
valueB*    *
_output_shapes
:
и
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
▀
coarse/fc2/fc2-b/Adam_1/AssignAssigncoarse/fc2/fc2-b/Adam_1)coarse/fc2/fc2-b/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
Л
coarse/fc2/fc2-b/Adam_1/readIdentitycoarse/fc2/fc2-b/Adam_1*#
_class
loc:@coarse/fc2/fc2-b*
T0*
_output_shapes
:
Я

Adam/beta1Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Я

Adam/beta2Const'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w╛?*
_output_shapes
: 
б
Adam/epsilonConst'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1*
dtype0*
valueB
 *w╠+2*
_output_shapes
: 
д
*Adam/update_coarse/conv1/conv1-w/ApplyAdam	ApplyAdamcoarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonNgradients/coarse/coarse/conv1/conv1/conv1-conv_grad/tuple/control_dependency_1*
use_nesterov( *'
_class
loc:@coarse/conv1/conv1-w*
use_locking( *
T0*&
_output_shapes
:
Ь
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
х
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
▌
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
р
+Adam/update_coarse/conv2-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_nesterov( *(
_class
loc:@coarse/conv2-bn/gamma*
use_locking( *
T0*
_output_shapes
: 
█
*Adam/update_coarse/conv2-bn/beta/ApplyAdam	ApplyAdamcoarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
use_nesterov( *'
_class
loc:@coarse/conv2-bn/beta*
use_locking( *
T0*
_output_shapes
: 
х
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
▌
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
▀
+Adam/update_coarse/conv3-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_nesterov( *(
_class
loc:@coarse/conv3-bn/gamma*
use_locking( *
T0*
_output_shapes
:@
┌
*Adam/update_coarse/conv3-bn/beta/ApplyAdam	ApplyAdamcoarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *'
_class
loc:@coarse/conv3-bn/beta*
use_locking( *
T0*
_output_shapes
:@
ц
4Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking( *
T0*'
_output_shapes
:@А
▐
4Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam	ApplyAdamcoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv4-conv/conv4-conv/conv4-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking( *
T0*
_output_shapes	
:А
р
+Adam/update_coarse/conv4-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_nesterov( *(
_class
loc:@coarse/conv4-bn/gamma*
use_locking( *
T0*
_output_shapes	
:А
█
*Adam/update_coarse/conv4-bn/beta/ApplyAdam	ApplyAdamcoarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( *'
_class
loc:@coarse/conv4-bn/beta*
use_locking( *
T0*
_output_shapes	
:А
ч
4Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam	ApplyAdamcoarse/conv5-conv/conv5-conv-w#coarse/conv5-conv/conv5-conv-w/Adam%coarse/conv5-conv/conv5-conv-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilon]gradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-conv_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking( *
T0*(
_output_shapes
:		АА
▐
4Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam	ApplyAdamcoarse/conv5-conv/conv5-conv-b#coarse/conv5-conv/conv5-conv-b/Adam%coarse/conv5-conv/conv5-conv-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonagradients/coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add_grad/tuple/control_dependency_1*
use_nesterov( *1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking( *
T0*
_output_shapes	
:А
р
+Adam/update_coarse/conv5-bn/gamma/ApplyAdam	ApplyAdamcoarse/conv5-bn/gammacoarse/conv5-bn/gamma/Adamcoarse/conv5-bn/gamma/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *(
_class
loc:@coarse/conv5-bn/gamma*
use_locking( *
T0*
_output_shapes	
:А
█
*Adam/update_coarse/conv5-bn/beta/ApplyAdam	ApplyAdamcoarse/conv5-bn/betacoarse/conv5-bn/beta/Adamcoarse/conv5-bn/beta/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *'
_class
loc:@coarse/conv5-bn/beta*
use_locking( *
T0*
_output_shapes	
:А
Д
&Adam/update_coarse/fc1/fc1-w/ApplyAdam	ApplyAdamcoarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc1/fc1/fc1-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-w*
use_locking( *
T0*!
_output_shapes
:АША
Г
&Adam/update_coarse/fc1/fc1-b/ApplyAdam	ApplyAdamcoarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonLgradients/coarse/coarse/fc1/fc1/fc1-bias_add_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc1/fc1-b*
use_locking( *
T0*
_output_shapes	
:А
В
&Adam/update_coarse/fc2/fc2-w/ApplyAdam	ApplyAdamcoarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1beta1_power/readbeta2_power/readlr
Adam/beta1
Adam/beta2Adam/epsilonGgradients/coarse/coarse/fc2/fc2/fc2-mat_grad/tuple/control_dependency_1*
use_nesterov( *#
_class
loc:@coarse/fc2/fc2-w*
use_locking( *
T0*
_output_shapes
:	А
В
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
Щ	
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
Я
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ы	

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam*'
_class
loc:@coarse/conv1/conv1-b*
T0*
_output_shapes
: 
г
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking( *
T0*
_output_shapes
: 
Ь
AdamNoOp'^coarse/coarse/conv2-bn/AssignMovingAvg)^coarse/coarse/conv2-bn/AssignMovingAvg_1'^coarse/coarse/conv3-bn/AssignMovingAvg)^coarse/coarse/conv3-bn/AssignMovingAvg_1'^coarse/coarse/conv4-bn/AssignMovingAvg)^coarse/coarse/conv4-bn/AssignMovingAvg_1'^coarse/coarse/conv5-bn/AssignMovingAvg)^coarse/coarse/conv5-bn/AssignMovingAvg_1+^Adam/update_coarse/conv1/conv1-w/ApplyAdam+^Adam/update_coarse/conv1/conv1-b/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-w/ApplyAdam5^Adam/update_coarse/conv2-conv/conv2-conv-b/ApplyAdam,^Adam/update_coarse/conv2-bn/gamma/ApplyAdam+^Adam/update_coarse/conv2-bn/beta/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-w/ApplyAdam5^Adam/update_coarse/conv3-conv/conv3-conv-b/ApplyAdam,^Adam/update_coarse/conv3-bn/gamma/ApplyAdam+^Adam/update_coarse/conv3-bn/beta/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-w/ApplyAdam5^Adam/update_coarse/conv4-conv/conv4-conv-b/ApplyAdam,^Adam/update_coarse/conv4-bn/gamma/ApplyAdam+^Adam/update_coarse/conv4-bn/beta/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-w/ApplyAdam5^Adam/update_coarse/conv5-conv/conv5-conv-b/ApplyAdam,^Adam/update_coarse/conv5-bn/gamma/ApplyAdam+^Adam/update_coarse/conv5-bn/beta/ApplyAdam'^Adam/update_coarse/fc1/fc1-w/ApplyAdam'^Adam/update_coarse/fc1/fc1-b/ApplyAdam'^Adam/update_coarse/fc2/fc2-w/ApplyAdam'^Adam/update_coarse/fc2/fc2-b/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Ї
save/SaveV2/tensor_namesConst*
dtype0*з
valueЭBЪLBbeta1_powerBbeta2_powerBcoarse/conv1/conv1-bBcoarse/conv1/conv1-b/AdamBcoarse/conv1/conv1-b/Adam_1Bcoarse/conv1/conv1-wBcoarse/conv1/conv1-w/AdamBcoarse/conv1/conv1-w/Adam_1Bcoarse/conv2-bn/betaBcoarse/conv2-bn/beta/AdamBcoarse/conv2-bn/beta/Adam_1Bcoarse/conv2-bn/gammaBcoarse/conv2-bn/gamma/AdamBcoarse/conv2-bn/gamma/Adam_1Bcoarse/conv2-bn/moving_meanBcoarse/conv2-bn/moving_varianceBcoarse/conv2-conv/conv2-conv-bB#coarse/conv2-conv/conv2-conv-b/AdamB%coarse/conv2-conv/conv2-conv-b/Adam_1Bcoarse/conv2-conv/conv2-conv-wB#coarse/conv2-conv/conv2-conv-w/AdamB%coarse/conv2-conv/conv2-conv-w/Adam_1Bcoarse/conv3-bn/betaBcoarse/conv3-bn/beta/AdamBcoarse/conv3-bn/beta/Adam_1Bcoarse/conv3-bn/gammaBcoarse/conv3-bn/gamma/AdamBcoarse/conv3-bn/gamma/Adam_1Bcoarse/conv3-bn/moving_meanBcoarse/conv3-bn/moving_varianceBcoarse/conv3-conv/conv3-conv-bB#coarse/conv3-conv/conv3-conv-b/AdamB%coarse/conv3-conv/conv3-conv-b/Adam_1Bcoarse/conv3-conv/conv3-conv-wB#coarse/conv3-conv/conv3-conv-w/AdamB%coarse/conv3-conv/conv3-conv-w/Adam_1Bcoarse/conv4-bn/betaBcoarse/conv4-bn/beta/AdamBcoarse/conv4-bn/beta/Adam_1Bcoarse/conv4-bn/gammaBcoarse/conv4-bn/gamma/AdamBcoarse/conv4-bn/gamma/Adam_1Bcoarse/conv4-bn/moving_meanBcoarse/conv4-bn/moving_varianceBcoarse/conv4-conv/conv4-conv-bB#coarse/conv4-conv/conv4-conv-b/AdamB%coarse/conv4-conv/conv4-conv-b/Adam_1Bcoarse/conv4-conv/conv4-conv-wB#coarse/conv4-conv/conv4-conv-w/AdamB%coarse/conv4-conv/conv4-conv-w/Adam_1Bcoarse/conv5-bn/betaBcoarse/conv5-bn/beta/AdamBcoarse/conv5-bn/beta/Adam_1Bcoarse/conv5-bn/gammaBcoarse/conv5-bn/gamma/AdamBcoarse/conv5-bn/gamma/Adam_1Bcoarse/conv5-bn/moving_meanBcoarse/conv5-bn/moving_varianceBcoarse/conv5-conv/conv5-conv-bB#coarse/conv5-conv/conv5-conv-b/AdamB%coarse/conv5-conv/conv5-conv-b/Adam_1Bcoarse/conv5-conv/conv5-conv-wB#coarse/conv5-conv/conv5-conv-w/AdamB%coarse/conv5-conv/conv5-conv-w/Adam_1Bcoarse/fc1/fc1-bBcoarse/fc1/fc1-b/AdamBcoarse/fc1/fc1-b/Adam_1Bcoarse/fc1/fc1-wBcoarse/fc1/fc1-w/AdamBcoarse/fc1/fc1-w/Adam_1Bcoarse/fc2/fc2-bBcoarse/fc2/fc2-b/AdamBcoarse/fc2/fc2-b/Adam_1Bcoarse/fc2/fc2-wBcoarse/fc2/fc2-w/AdamBcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:L
■
save/SaveV2/shape_and_slicesConst*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
╟
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powercoarse/conv1/conv1-bcoarse/conv1/conv1-b/Adamcoarse/conv1/conv1-b/Adam_1coarse/conv1/conv1-wcoarse/conv1/conv1-w/Adamcoarse/conv1/conv1-w/Adam_1coarse/conv2-bn/betacoarse/conv2-bn/beta/Adamcoarse/conv2-bn/beta/Adam_1coarse/conv2-bn/gammacoarse/conv2-bn/gamma/Adamcoarse/conv2-bn/gamma/Adam_1coarse/conv2-bn/moving_meancoarse/conv2-bn/moving_variancecoarse/conv2-conv/conv2-conv-b#coarse/conv2-conv/conv2-conv-b/Adam%coarse/conv2-conv/conv2-conv-b/Adam_1coarse/conv2-conv/conv2-conv-w#coarse/conv2-conv/conv2-conv-w/Adam%coarse/conv2-conv/conv2-conv-w/Adam_1coarse/conv3-bn/betacoarse/conv3-bn/beta/Adamcoarse/conv3-bn/beta/Adam_1coarse/conv3-bn/gammacoarse/conv3-bn/gamma/Adamcoarse/conv3-bn/gamma/Adam_1coarse/conv3-bn/moving_meancoarse/conv3-bn/moving_variancecoarse/conv3-conv/conv3-conv-b#coarse/conv3-conv/conv3-conv-b/Adam%coarse/conv3-conv/conv3-conv-b/Adam_1coarse/conv3-conv/conv3-conv-w#coarse/conv3-conv/conv3-conv-w/Adam%coarse/conv3-conv/conv3-conv-w/Adam_1coarse/conv4-bn/betacoarse/conv4-bn/beta/Adamcoarse/conv4-bn/beta/Adam_1coarse/conv4-bn/gammacoarse/conv4-bn/gamma/Adamcoarse/conv4-bn/gamma/Adam_1coarse/conv4-bn/moving_meancoarse/conv4-bn/moving_variancecoarse/conv4-conv/conv4-conv-b#coarse/conv4-conv/conv4-conv-b/Adam%coarse/conv4-conv/conv4-conv-b/Adam_1coarse/conv4-conv/conv4-conv-w#coarse/conv4-conv/conv4-conv-w/Adam%coarse/conv4-conv/conv4-conv-w/Adam_1coarse/conv5-bn/betacoarse/conv5-bn/beta/Adamcoarse/conv5-bn/beta/Adam_1coarse/conv5-bn/gammacoarse/conv5-bn/gamma/Adamcoarse/conv5-bn/gamma/Adam_1coarse/conv5-bn/moving_meancoarse/conv5-bn/moving_variancecoarse/conv5-conv/conv5-conv-b#coarse/conv5-conv/conv5-conv-b/Adam%coarse/conv5-conv/conv5-conv-b/Adam_1coarse/conv5-conv/conv5-conv-w#coarse/conv5-conv/conv5-conv-w/Adam%coarse/conv5-conv/conv5-conv-w/Adam_1coarse/fc1/fc1-bcoarse/fc1/fc1-b/Adamcoarse/fc1/fc1-b/Adam_1coarse/fc1/fc1-wcoarse/fc1/fc1-w/Adamcoarse/fc1/fc1-w/Adam_1coarse/fc2/fc2-bcoarse/fc2/fc2-b/Adamcoarse/fc2/fc2-b/Adam_1coarse/fc2/fc2-wcoarse/fc2/fc2-w/Adamcoarse/fc2/fc2-w/Adam_1*Z
dtypesP
N2L
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
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
е
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
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
й
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
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
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
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_3Assigncoarse/conv1/conv1-b/Adamsave/RestoreV2_3*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-b*
use_locking(*
T0*
_output_shapes
:
Б
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
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╜
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
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_5Assigncoarse/conv1/conv1-wsave/RestoreV2_5*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
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
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_6Assigncoarse/conv1/conv1-w/Adamsave/RestoreV2_6*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
Б
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
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
╔
save/Assign_7Assigncoarse/conv1/conv1-w/Adam_1save/RestoreV2_7*
validate_shape(*'
_class
loc:@coarse/conv1/conv1-w*
use_locking(*
T0*&
_output_shapes
:
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
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
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
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_9Assigncoarse/conv2-bn/beta/Adamsave/RestoreV2_9*
validate_shape(*'
_class
loc:@coarse/conv2-bn/beta*
use_locking(*
T0*
_output_shapes
: 
В
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
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
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
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_11Assigncoarse/conv2-bn/gammasave/RestoreV2_11*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Б
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
Щ
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
save/Assign_12Assigncoarse/conv2-bn/gamma/Adamsave/RestoreV2_12*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
Г
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
Щ
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
┴
save/Assign_13Assigncoarse/conv2-bn/gamma/Adam_1save/RestoreV2_13*
validate_shape(*(
_class
loc:@coarse/conv2-bn/gamma*
use_locking(*
T0*
_output_shapes
: 
В
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
Щ
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
╞
save/Assign_14Assigncoarse/conv2-bn/moving_meansave/RestoreV2_14*
validate_shape(*.
_class$
" loc:@coarse/conv2-bn/moving_mean*
use_locking(*
T0*
_output_shapes
: 
Ж
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
Щ
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
╬
save/Assign_15Assigncoarse/conv2-bn/moving_variancesave/RestoreV2_15*
validate_shape(*2
_class(
&$loc:@coarse/conv2-bn/moving_variance*
use_locking(*
T0*
_output_shapes
: 
Е
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
Щ
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_16Assigncoarse/conv2-conv/conv2-conv-bsave/RestoreV2_16*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
К
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
Щ
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
╤
save/Assign_17Assign#coarse/conv2-conv/conv2-conv-b/Adamsave/RestoreV2_17*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
М
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
Щ
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_18Assign%coarse/conv2-conv/conv2-conv-b/Adam_1save/RestoreV2_18*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-b*
use_locking(*
T0*
_output_shapes
: 
Е
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
Щ
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
╪
save/Assign_19Assigncoarse/conv2-conv/conv2-conv-wsave/RestoreV2_19*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
К
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
Щ
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
▌
save/Assign_20Assign#coarse/conv2-conv/conv2-conv-w/Adamsave/RestoreV2_20*
validate_shape(*1
_class'
%#loc:@coarse/conv2-conv/conv2-conv-w*
use_locking(*
T0*&
_output_shapes
: 
М
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
Щ
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
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
Щ
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_22Assigncoarse/conv3-bn/betasave/RestoreV2_22*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
А
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
Щ
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
╜
save/Assign_23Assigncoarse/conv3-bn/beta/Adamsave/RestoreV2_23*
validate_shape(*'
_class
loc:@coarse/conv3-bn/beta*
use_locking(*
T0*
_output_shapes
:@
В
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
Щ
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
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
Щ
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_25Assigncoarse/conv3-bn/gammasave/RestoreV2_25*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Б
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
Щ
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
┐
save/Assign_26Assigncoarse/conv3-bn/gamma/Adamsave/RestoreV2_26*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
Г
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
Щ
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
┴
save/Assign_27Assigncoarse/conv3-bn/gamma/Adam_1save/RestoreV2_27*
validate_shape(*(
_class
loc:@coarse/conv3-bn/gamma*
use_locking(*
T0*
_output_shapes
:@
В
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
Щ
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
╞
save/Assign_28Assigncoarse/conv3-bn/moving_meansave/RestoreV2_28*
validate_shape(*.
_class$
" loc:@coarse/conv3-bn/moving_mean*
use_locking(*
T0*
_output_shapes
:@
Ж
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
Щ
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
╬
save/Assign_29Assigncoarse/conv3-bn/moving_variancesave/RestoreV2_29*
validate_shape(*2
_class(
&$loc:@coarse/conv3-bn/moving_variance*
use_locking(*
T0*
_output_shapes
:@
Е
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
Щ
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_30Assigncoarse/conv3-conv/conv3-conv-bsave/RestoreV2_30*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
К
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
Щ
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
╤
save/Assign_31Assign#coarse/conv3-conv/conv3-conv-b/Adamsave/RestoreV2_31*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
М
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
Щ
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_32Assign%coarse/conv3-conv/conv3-conv-b/Adam_1save/RestoreV2_32*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-b*
use_locking(*
T0*
_output_shapes
:@
Е
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
Щ
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
╪
save/Assign_33Assigncoarse/conv3-conv/conv3-conv-wsave/RestoreV2_33*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
К
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
Щ
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
▌
save/Assign_34Assign#coarse/conv3-conv/conv3-conv-w/Adamsave/RestoreV2_34*
validate_shape(*1
_class'
%#loc:@coarse/conv3-conv/conv3-conv-w*
use_locking(*
T0*&
_output_shapes
: @
М
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
Щ
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
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
Щ
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
╣
save/Assign_36Assigncoarse/conv4-bn/betasave/RestoreV2_36*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
А
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
Щ
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_37Assigncoarse/conv4-bn/beta/Adamsave/RestoreV2_37*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
В
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
Щ
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_38Assigncoarse/conv4-bn/beta/Adam_1save/RestoreV2_38*
validate_shape(*'
_class
loc:@coarse/conv4-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
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
Щ
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_39Assigncoarse/conv4-bn/gammasave/RestoreV2_39*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Б
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
Щ
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_40Assigncoarse/conv4-bn/gamma/Adamsave/RestoreV2_40*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Г
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
Щ
save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_41Assigncoarse/conv4-bn/gamma/Adam_1save/RestoreV2_41*
validate_shape(*(
_class
loc:@coarse/conv4-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
В
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
Щ
save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_42Assigncoarse/conv4-bn/moving_meansave/RestoreV2_42*
validate_shape(*.
_class$
" loc:@coarse/conv4-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Ж
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
Щ
save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
╧
save/Assign_43Assigncoarse/conv4-bn/moving_variancesave/RestoreV2_43*
validate_shape(*2
_class(
&$loc:@coarse/conv4-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
Е
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
Щ
save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
═
save/Assign_44Assigncoarse/conv4-conv/conv4-conv-bsave/RestoreV2_44*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
К
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
Щ
save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_45Assign#coarse/conv4-conv/conv4-conv-b/Adamsave/RestoreV2_45*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
М
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
Щ
save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
╘
save/Assign_46Assign%coarse/conv4-conv/conv4-conv-b/Adam_1save/RestoreV2_46*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-b*
use_locking(*
T0*
_output_shapes	
:А
Е
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
Щ
save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
┘
save/Assign_47Assigncoarse/conv4-conv/conv4-conv-wsave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
К
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
Щ
save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
▐
save/Assign_48Assign#coarse/conv4-conv/conv4-conv-w/Adamsave/RestoreV2_48*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
М
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
Щ
save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
р
save/Assign_49Assign%coarse/conv4-conv/conv4-conv-w/Adam_1save/RestoreV2_49*
validate_shape(*1
_class'
%#loc:@coarse/conv4-conv/conv4-conv-w*
use_locking(*
T0*'
_output_shapes
:@А
{
save/RestoreV2_50/tensor_namesConst*
dtype0*)
value BBcoarse/conv5-bn/beta*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
╣
save/Assign_50Assigncoarse/conv5-bn/betasave/RestoreV2_50*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
А
save/RestoreV2_51/tensor_namesConst*
dtype0*.
value%B#Bcoarse/conv5-bn/beta/Adam*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_51Assigncoarse/conv5-bn/beta/Adamsave/RestoreV2_51*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
В
save/RestoreV2_52/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv5-bn/beta/Adam_1*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_52Assigncoarse/conv5-bn/beta/Adam_1save/RestoreV2_52*
validate_shape(*'
_class
loc:@coarse/conv5-bn/beta*
use_locking(*
T0*
_output_shapes	
:А
|
save/RestoreV2_53/tensor_namesConst*
dtype0**
value!BBcoarse/conv5-bn/gamma*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
╗
save/Assign_53Assigncoarse/conv5-bn/gammasave/RestoreV2_53*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Б
save/RestoreV2_54/tensor_namesConst*
dtype0*/
value&B$Bcoarse/conv5-bn/gamma/Adam*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
└
save/Assign_54Assigncoarse/conv5-bn/gamma/Adamsave/RestoreV2_54*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
Г
save/RestoreV2_55/tensor_namesConst*
dtype0*1
value(B&Bcoarse/conv5-bn/gamma/Adam_1*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save/Assign_55Assigncoarse/conv5-bn/gamma/Adam_1save/RestoreV2_55*
validate_shape(*(
_class
loc:@coarse/conv5-bn/gamma*
use_locking(*
T0*
_output_shapes	
:А
В
save/RestoreV2_56/tensor_namesConst*
dtype0*0
value'B%Bcoarse/conv5-bn/moving_mean*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
╟
save/Assign_56Assigncoarse/conv5-bn/moving_meansave/RestoreV2_56*
validate_shape(*.
_class$
" loc:@coarse/conv5-bn/moving_mean*
use_locking(*
T0*
_output_shapes	
:А
Ж
save/RestoreV2_57/tensor_namesConst*
dtype0*4
value+B)Bcoarse/conv5-bn/moving_variance*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
╧
save/Assign_57Assigncoarse/conv5-bn/moving_variancesave/RestoreV2_57*
validate_shape(*2
_class(
&$loc:@coarse/conv5-bn/moving_variance*
use_locking(*
T0*
_output_shapes	
:А
Е
save/RestoreV2_58/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv5-conv/conv5-conv-b*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
═
save/Assign_58Assigncoarse/conv5-conv/conv5-conv-bsave/RestoreV2_58*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
К
save/RestoreV2_59/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv5-conv/conv5-conv-b/Adam*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_59Assign#coarse/conv5-conv/conv5-conv-b/Adamsave/RestoreV2_59*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
М
save/RestoreV2_60/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv5-conv/conv5-conv-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_60/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
╘
save/Assign_60Assign%coarse/conv5-conv/conv5-conv-b/Adam_1save/RestoreV2_60*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-b*
use_locking(*
T0*
_output_shapes	
:А
Е
save/RestoreV2_61/tensor_namesConst*
dtype0*3
value*B(Bcoarse/conv5-conv/conv5-conv-w*
_output_shapes
:
k
"save/RestoreV2_61/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
┌
save/Assign_61Assigncoarse/conv5-conv/conv5-conv-wsave/RestoreV2_61*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
К
save/RestoreV2_62/tensor_namesConst*
dtype0*8
value/B-B#coarse/conv5-conv/conv5-conv-w/Adam*
_output_shapes
:
k
"save/RestoreV2_62/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_62	RestoreV2
save/Constsave/RestoreV2_62/tensor_names"save/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
▀
save/Assign_62Assign#coarse/conv5-conv/conv5-conv-w/Adamsave/RestoreV2_62*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
М
save/RestoreV2_63/tensor_namesConst*
dtype0*:
value1B/B%coarse/conv5-conv/conv5-conv-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_63/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_63	RestoreV2
save/Constsave/RestoreV2_63/tensor_names"save/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
с
save/Assign_63Assign%coarse/conv5-conv/conv5-conv-w/Adam_1save/RestoreV2_63*
validate_shape(*1
_class'
%#loc:@coarse/conv5-conv/conv5-conv-w*
use_locking(*
T0*(
_output_shapes
:		АА
w
save/RestoreV2_64/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-b*
_output_shapes
:
k
"save/RestoreV2_64/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_64	RestoreV2
save/Constsave/RestoreV2_64/tensor_names"save/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
▒
save/Assign_64Assigncoarse/fc1/fc1-bsave/RestoreV2_64*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
|
save/RestoreV2_65/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-b/Adam*
_output_shapes
:
k
"save/RestoreV2_65/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_65	RestoreV2
save/Constsave/RestoreV2_65/tensor_names"save/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
╢
save/Assign_65Assigncoarse/fc1/fc1-b/Adamsave/RestoreV2_65*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
~
save/RestoreV2_66/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_66/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_66	RestoreV2
save/Constsave/RestoreV2_66/tensor_names"save/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
╕
save/Assign_66Assigncoarse/fc1/fc1-b/Adam_1save/RestoreV2_66*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-b*
use_locking(*
T0*
_output_shapes	
:А
w
save/RestoreV2_67/tensor_namesConst*
dtype0*%
valueBBcoarse/fc1/fc1-w*
_output_shapes
:
k
"save/RestoreV2_67/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_67	RestoreV2
save/Constsave/RestoreV2_67/tensor_names"save/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
╖
save/Assign_67Assigncoarse/fc1/fc1-wsave/RestoreV2_67*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
|
save/RestoreV2_68/tensor_namesConst*
dtype0**
value!BBcoarse/fc1/fc1-w/Adam*
_output_shapes
:
k
"save/RestoreV2_68/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_68	RestoreV2
save/Constsave/RestoreV2_68/tensor_names"save/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
╝
save/Assign_68Assigncoarse/fc1/fc1-w/Adamsave/RestoreV2_68*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
~
save/RestoreV2_69/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc1/fc1-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_69/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_69	RestoreV2
save/Constsave/RestoreV2_69/tensor_names"save/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_69Assigncoarse/fc1/fc1-w/Adam_1save/RestoreV2_69*
validate_shape(*#
_class
loc:@coarse/fc1/fc1-w*
use_locking(*
T0*!
_output_shapes
:АША
w
save/RestoreV2_70/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-b*
_output_shapes
:
k
"save/RestoreV2_70/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_70	RestoreV2
save/Constsave/RestoreV2_70/tensor_names"save/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
░
save/Assign_70Assigncoarse/fc2/fc2-bsave/RestoreV2_70*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
|
save/RestoreV2_71/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-b/Adam*
_output_shapes
:
k
"save/RestoreV2_71/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_71	RestoreV2
save/Constsave/RestoreV2_71/tensor_names"save/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
╡
save/Assign_71Assigncoarse/fc2/fc2-b/Adamsave/RestoreV2_71*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
~
save/RestoreV2_72/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-b/Adam_1*
_output_shapes
:
k
"save/RestoreV2_72/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_72	RestoreV2
save/Constsave/RestoreV2_72/tensor_names"save/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
╖
save/Assign_72Assigncoarse/fc2/fc2-b/Adam_1save/RestoreV2_72*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-b*
use_locking(*
T0*
_output_shapes
:
w
save/RestoreV2_73/tensor_namesConst*
dtype0*%
valueBBcoarse/fc2/fc2-w*
_output_shapes
:
k
"save/RestoreV2_73/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_73	RestoreV2
save/Constsave/RestoreV2_73/tensor_names"save/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
╡
save/Assign_73Assigncoarse/fc2/fc2-wsave/RestoreV2_73*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
|
save/RestoreV2_74/tensor_namesConst*
dtype0**
value!BBcoarse/fc2/fc2-w/Adam*
_output_shapes
:
k
"save/RestoreV2_74/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_74	RestoreV2
save/Constsave/RestoreV2_74/tensor_names"save/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
║
save/Assign_74Assigncoarse/fc2/fc2-w/Adamsave/RestoreV2_74*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
~
save/RestoreV2_75/tensor_namesConst*
dtype0*,
value#B!Bcoarse/fc2/fc2-w/Adam_1*
_output_shapes
:
k
"save/RestoreV2_75/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Щ
save/RestoreV2_75	RestoreV2
save/Constsave/RestoreV2_75/tensor_names"save/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
╝
save/Assign_75Assigncoarse/fc2/fc2-w/Adam_1save/RestoreV2_75*
validate_shape(*#
_class
loc:@coarse/fc2/fc2-w*
use_locking(*
T0*
_output_shapes
:	А
Ш

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""ЇM
cond_contextуMрM
┤
%coarse/coarse/conv2-bn/cond/cond_text%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_t:0 *╣
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
■

'coarse/coarse/conv2-bn/cond/cond_text_1%coarse/coarse/conv2-bn/cond/pred_id:0&coarse/coarse/conv2-bn/cond/switch_f:0*Г

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
┤
%coarse/coarse/conv3-bn/cond/cond_text%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_t:0 *╣
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
■

'coarse/coarse/conv3-bn/cond/cond_text_1%coarse/coarse/conv3-bn/cond/pred_id:0&coarse/coarse/conv3-bn/cond/switch_f:0*Г

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
┤
%coarse/coarse/conv4-bn/cond/cond_text%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_t:0 *╣
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
■

'coarse/coarse/conv4-bn/cond/cond_text_1%coarse/coarse/conv4-bn/cond/pred_id:0&coarse/coarse/conv4-bn/cond/switch_f:0*Г

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
&coarse/conv4-bn/moving_variance/read:07coarse/coarse/conv4-bn/cond/FusedBatchNorm_1/Switch_4:0
┤
%coarse/coarse/conv5-bn/cond/cond_text%coarse/coarse/conv5-bn/cond/pred_id:0&coarse/coarse/conv5-bn/cond/switch_t:0 *╣
#coarse/coarse/conv5-bn/cond/Const:0
%coarse/coarse/conv5-bn/cond/Const_1:0
3coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:1
5coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:1
5coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2:1
,coarse/coarse/conv5-bn/cond/FusedBatchNorm:0
,coarse/coarse/conv5-bn/cond/FusedBatchNorm:1
,coarse/coarse/conv5-bn/cond/FusedBatchNorm:2
,coarse/coarse/conv5-bn/cond/FusedBatchNorm:3
,coarse/coarse/conv5-bn/cond/FusedBatchNorm:4
%coarse/coarse/conv5-bn/cond/pred_id:0
&coarse/coarse/conv5-bn/cond/switch_t:0
9coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add:0
coarse/conv5-bn/beta/read:0
coarse/conv5-bn/gamma/read:0U
coarse/conv5-bn/gamma/read:05coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_1:1T
coarse/conv5-bn/beta/read:05coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch_2:1p
9coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add:03coarse/coarse/conv5-bn/cond/FusedBatchNorm/Switch:1
■

'coarse/coarse/conv5-bn/cond/cond_text_1%coarse/coarse/conv5-bn/cond/pred_id:0&coarse/coarse/conv5-bn/cond/switch_f:0*Г

5coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch:0
7coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1:0
7coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2:0
7coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_3:0
7coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4:0
.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:0
.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:1
.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:2
.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:3
.coarse/coarse/conv5-bn/cond/FusedBatchNorm_1:4
%coarse/coarse/conv5-bn/cond/pred_id:0
&coarse/coarse/conv5-bn/cond/switch_f:0
9coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add:0
coarse/conv5-bn/beta/read:0
coarse/conv5-bn/gamma/read:0
"coarse/conv5-bn/moving_mean/read:0
&coarse/conv5-bn/moving_variance/read:0W
coarse/conv5-bn/gamma/read:07coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_1:0V
coarse/conv5-bn/beta/read:07coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_2:0]
"coarse/conv5-bn/moving_mean/read:07coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_3:0a
&coarse/conv5-bn/moving_variance/read:07coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch_4:0r
9coarse/coarse/conv5-conv/conv5-conv/conv5-conv-biad_add:05coarse/coarse/conv5-bn/cond/FusedBatchNorm_1/Switch:0"╤
trainable_variables╣╢
З
coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
п
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
д
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0
п
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
д
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0
п
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
д
 coarse/conv4-conv/conv4-conv-b:0%coarse/conv4-conv/conv4-conv-b/Assign%coarse/conv4-conv/conv4-conv-b/read:022coarse/conv4-conv/conv4-conv-b/Initializer/Const:0

coarse/conv4-bn/gamma:0coarse/conv4-bn/gamma/Assigncoarse/conv4-bn/gamma/read:02(coarse/conv4-bn/gamma/Initializer/ones:0
|
coarse/conv4-bn/beta:0coarse/conv4-bn/beta/Assigncoarse/conv4-bn/beta/read:02(coarse/conv4-bn/beta/Initializer/zeros:0
п
 coarse/conv5-conv/conv5-conv-w:0%coarse/conv5-conv/conv5-conv-w/Assign%coarse/conv5-conv/conv5-conv-w/read:02=coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal:0
д
 coarse/conv5-conv/conv5-conv-b:0%coarse/conv5-conv/conv5-conv-b/Assign%coarse/conv5-conv/conv5-conv-b/read:022coarse/conv5-conv/conv5-conv-b/Initializer/Const:0

coarse/conv5-bn/gamma:0coarse/conv5-bn/gamma/Assigncoarse/conv5-bn/gamma/read:02(coarse/conv5-bn/gamma/Initializer/ones:0
|
coarse/conv5-bn/beta:0coarse/conv5-bn/beta/Assigncoarse/conv5-bn/beta/read:02(coarse/conv5-bn/beta/Initializer/zeros:0
w
coarse/fc1/fc1-w:0coarse/fc1/fc1-w/Assigncoarse/fc1/fc1-w/read:02/coarse/fc1/fc1-w/Initializer/truncated_normal:0
l
coarse/fc1/fc1-b:0coarse/fc1/fc1-b/Assigncoarse/fc1/fc1-b/read:02$coarse/fc1/fc1-b/Initializer/Const:0
w
coarse/fc2/fc2-w:0coarse/fc2/fc2-w/Assigncoarse/fc2/fc2-w/read:02/coarse/fc2/fc2-w/Initializer/truncated_normal:0
l
coarse/fc2/fc2-b:0coarse/fc2/fc2-b/Assigncoarse/fc2/fc2-b/read:02$coarse/fc2/fc2-b/Initializer/Const:0"Ы\
	variablesН\К\
З
coarse/conv1/conv1-w:0coarse/conv1/conv1-w/Assigncoarse/conv1/conv1-w/read:023coarse/conv1/conv1-w/Initializer/truncated_normal:0
|
coarse/conv1/conv1-b:0coarse/conv1/conv1-b/Assigncoarse/conv1/conv1-b/read:02(coarse/conv1/conv1-b/Initializer/Const:0
п
 coarse/conv2-conv/conv2-conv-w:0%coarse/conv2-conv/conv2-conv-w/Assign%coarse/conv2-conv/conv2-conv-w/read:02=coarse/conv2-conv/conv2-conv-w/Initializer/truncated_normal:0
д
 coarse/conv2-conv/conv2-conv-b:0%coarse/conv2-conv/conv2-conv-b/Assign%coarse/conv2-conv/conv2-conv-b/read:022coarse/conv2-conv/conv2-conv-b/Initializer/Const:0

coarse/conv2-bn/gamma:0coarse/conv2-bn/gamma/Assigncoarse/conv2-bn/gamma/read:02(coarse/conv2-bn/gamma/Initializer/ones:0
|
coarse/conv2-bn/beta:0coarse/conv2-bn/beta/Assigncoarse/conv2-bn/beta/read:02(coarse/conv2-bn/beta/Initializer/zeros:0
Ш
coarse/conv2-bn/moving_mean:0"coarse/conv2-bn/moving_mean/Assign"coarse/conv2-bn/moving_mean/read:02/coarse/conv2-bn/moving_mean/Initializer/zeros:0
з
!coarse/conv2-bn/moving_variance:0&coarse/conv2-bn/moving_variance/Assign&coarse/conv2-bn/moving_variance/read:022coarse/conv2-bn/moving_variance/Initializer/ones:0
п
 coarse/conv3-conv/conv3-conv-w:0%coarse/conv3-conv/conv3-conv-w/Assign%coarse/conv3-conv/conv3-conv-w/read:02=coarse/conv3-conv/conv3-conv-w/Initializer/truncated_normal:0
д
 coarse/conv3-conv/conv3-conv-b:0%coarse/conv3-conv/conv3-conv-b/Assign%coarse/conv3-conv/conv3-conv-b/read:022coarse/conv3-conv/conv3-conv-b/Initializer/Const:0

coarse/conv3-bn/gamma:0coarse/conv3-bn/gamma/Assigncoarse/conv3-bn/gamma/read:02(coarse/conv3-bn/gamma/Initializer/ones:0
|
coarse/conv3-bn/beta:0coarse/conv3-bn/beta/Assigncoarse/conv3-bn/beta/read:02(coarse/conv3-bn/beta/Initializer/zeros:0
Ш
coarse/conv3-bn/moving_mean:0"coarse/conv3-bn/moving_mean/Assign"coarse/conv3-bn/moving_mean/read:02/coarse/conv3-bn/moving_mean/Initializer/zeros:0
з
!coarse/conv3-bn/moving_variance:0&coarse/conv3-bn/moving_variance/Assign&coarse/conv3-bn/moving_variance/read:022coarse/conv3-bn/moving_variance/Initializer/ones:0
п
 coarse/conv4-conv/conv4-conv-w:0%coarse/conv4-conv/conv4-conv-w/Assign%coarse/conv4-conv/conv4-conv-w/read:02=coarse/conv4-conv/conv4-conv-w/Initializer/truncated_normal:0
д
 coarse/conv4-conv/conv4-conv-b:0%coarse/conv4-conv/conv4-conv-b/Assign%coarse/conv4-conv/conv4-conv-b/read:022coarse/conv4-conv/conv4-conv-b/Initializer/Const:0

coarse/conv4-bn/gamma:0coarse/conv4-bn/gamma/Assigncoarse/conv4-bn/gamma/read:02(coarse/conv4-bn/gamma/Initializer/ones:0
|
coarse/conv4-bn/beta:0coarse/conv4-bn/beta/Assigncoarse/conv4-bn/beta/read:02(coarse/conv4-bn/beta/Initializer/zeros:0
Ш
coarse/conv4-bn/moving_mean:0"coarse/conv4-bn/moving_mean/Assign"coarse/conv4-bn/moving_mean/read:02/coarse/conv4-bn/moving_mean/Initializer/zeros:0
з
!coarse/conv4-bn/moving_variance:0&coarse/conv4-bn/moving_variance/Assign&coarse/conv4-bn/moving_variance/read:022coarse/conv4-bn/moving_variance/Initializer/ones:0
п
 coarse/conv5-conv/conv5-conv-w:0%coarse/conv5-conv/conv5-conv-w/Assign%coarse/conv5-conv/conv5-conv-w/read:02=coarse/conv5-conv/conv5-conv-w/Initializer/truncated_normal:0
д
 coarse/conv5-conv/conv5-conv-b:0%coarse/conv5-conv/conv5-conv-b/Assign%coarse/conv5-conv/conv5-conv-b/read:022coarse/conv5-conv/conv5-conv-b/Initializer/Const:0

coarse/conv5-bn/gamma:0coarse/conv5-bn/gamma/Assigncoarse/conv5-bn/gamma/read:02(coarse/conv5-bn/gamma/Initializer/ones:0
|
coarse/conv5-bn/beta:0coarse/conv5-bn/beta/Assigncoarse/conv5-bn/beta/read:02(coarse/conv5-bn/beta/Initializer/zeros:0
Ш
coarse/conv5-bn/moving_mean:0"coarse/conv5-bn/moving_mean/Assign"coarse/conv5-bn/moving_mean/read:02/coarse/conv5-bn/moving_mean/Initializer/zeros:0
з
!coarse/conv5-bn/moving_variance:0&coarse/conv5-bn/moving_variance/Assign&coarse/conv5-bn/moving_variance/read:022coarse/conv5-bn/moving_variance/Initializer/ones:0
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
Р
coarse/conv1/conv1-w/Adam:0 coarse/conv1/conv1-w/Adam/Assign coarse/conv1/conv1-w/Adam/read:02-coarse/conv1/conv1-w/Adam/Initializer/zeros:0
Ш
coarse/conv1/conv1-w/Adam_1:0"coarse/conv1/conv1-w/Adam_1/Assign"coarse/conv1/conv1-w/Adam_1/read:02/coarse/conv1/conv1-w/Adam_1/Initializer/zeros:0
Р
coarse/conv1/conv1-b/Adam:0 coarse/conv1/conv1-b/Adam/Assign coarse/conv1/conv1-b/Adam/read:02-coarse/conv1/conv1-b/Adam/Initializer/zeros:0
Ш
coarse/conv1/conv1-b/Adam_1:0"coarse/conv1/conv1-b/Adam_1/Assign"coarse/conv1/conv1-b/Adam_1/read:02/coarse/conv1/conv1-b/Adam_1/Initializer/zeros:0
╕
%coarse/conv2-conv/conv2-conv-w/Adam:0*coarse/conv2-conv/conv2-conv-w/Adam/Assign*coarse/conv2-conv/conv2-conv-w/Adam/read:027coarse/conv2-conv/conv2-conv-w/Adam/Initializer/zeros:0
└
'coarse/conv2-conv/conv2-conv-w/Adam_1:0,coarse/conv2-conv/conv2-conv-w/Adam_1/Assign,coarse/conv2-conv/conv2-conv-w/Adam_1/read:029coarse/conv2-conv/conv2-conv-w/Adam_1/Initializer/zeros:0
╕
%coarse/conv2-conv/conv2-conv-b/Adam:0*coarse/conv2-conv/conv2-conv-b/Adam/Assign*coarse/conv2-conv/conv2-conv-b/Adam/read:027coarse/conv2-conv/conv2-conv-b/Adam/Initializer/zeros:0
└
'coarse/conv2-conv/conv2-conv-b/Adam_1:0,coarse/conv2-conv/conv2-conv-b/Adam_1/Assign,coarse/conv2-conv/conv2-conv-b/Adam_1/read:029coarse/conv2-conv/conv2-conv-b/Adam_1/Initializer/zeros:0
Ф
coarse/conv2-bn/gamma/Adam:0!coarse/conv2-bn/gamma/Adam/Assign!coarse/conv2-bn/gamma/Adam/read:02.coarse/conv2-bn/gamma/Adam/Initializer/zeros:0
Ь
coarse/conv2-bn/gamma/Adam_1:0#coarse/conv2-bn/gamma/Adam_1/Assign#coarse/conv2-bn/gamma/Adam_1/read:020coarse/conv2-bn/gamma/Adam_1/Initializer/zeros:0
Р
coarse/conv2-bn/beta/Adam:0 coarse/conv2-bn/beta/Adam/Assign coarse/conv2-bn/beta/Adam/read:02-coarse/conv2-bn/beta/Adam/Initializer/zeros:0
Ш
coarse/conv2-bn/beta/Adam_1:0"coarse/conv2-bn/beta/Adam_1/Assign"coarse/conv2-bn/beta/Adam_1/read:02/coarse/conv2-bn/beta/Adam_1/Initializer/zeros:0
╕
%coarse/conv3-conv/conv3-conv-w/Adam:0*coarse/conv3-conv/conv3-conv-w/Adam/Assign*coarse/conv3-conv/conv3-conv-w/Adam/read:027coarse/conv3-conv/conv3-conv-w/Adam/Initializer/zeros:0
└
'coarse/conv3-conv/conv3-conv-w/Adam_1:0,coarse/conv3-conv/conv3-conv-w/Adam_1/Assign,coarse/conv3-conv/conv3-conv-w/Adam_1/read:029coarse/conv3-conv/conv3-conv-w/Adam_1/Initializer/zeros:0
╕
%coarse/conv3-conv/conv3-conv-b/Adam:0*coarse/conv3-conv/conv3-conv-b/Adam/Assign*coarse/conv3-conv/conv3-conv-b/Adam/read:027coarse/conv3-conv/conv3-conv-b/Adam/Initializer/zeros:0
└
'coarse/conv3-conv/conv3-conv-b/Adam_1:0,coarse/conv3-conv/conv3-conv-b/Adam_1/Assign,coarse/conv3-conv/conv3-conv-b/Adam_1/read:029coarse/conv3-conv/conv3-conv-b/Adam_1/Initializer/zeros:0
Ф
coarse/conv3-bn/gamma/Adam:0!coarse/conv3-bn/gamma/Adam/Assign!coarse/conv3-bn/gamma/Adam/read:02.coarse/conv3-bn/gamma/Adam/Initializer/zeros:0
Ь
coarse/conv3-bn/gamma/Adam_1:0#coarse/conv3-bn/gamma/Adam_1/Assign#coarse/conv3-bn/gamma/Adam_1/read:020coarse/conv3-bn/gamma/Adam_1/Initializer/zeros:0
Р
coarse/conv3-bn/beta/Adam:0 coarse/conv3-bn/beta/Adam/Assign coarse/conv3-bn/beta/Adam/read:02-coarse/conv3-bn/beta/Adam/Initializer/zeros:0
Ш
coarse/conv3-bn/beta/Adam_1:0"coarse/conv3-bn/beta/Adam_1/Assign"coarse/conv3-bn/beta/Adam_1/read:02/coarse/conv3-bn/beta/Adam_1/Initializer/zeros:0
╕
%coarse/conv4-conv/conv4-conv-w/Adam:0*coarse/conv4-conv/conv4-conv-w/Adam/Assign*coarse/conv4-conv/conv4-conv-w/Adam/read:027coarse/conv4-conv/conv4-conv-w/Adam/Initializer/zeros:0
└
'coarse/conv4-conv/conv4-conv-w/Adam_1:0,coarse/conv4-conv/conv4-conv-w/Adam_1/Assign,coarse/conv4-conv/conv4-conv-w/Adam_1/read:029coarse/conv4-conv/conv4-conv-w/Adam_1/Initializer/zeros:0
╕
%coarse/conv4-conv/conv4-conv-b/Adam:0*coarse/conv4-conv/conv4-conv-b/Adam/Assign*coarse/conv4-conv/conv4-conv-b/Adam/read:027coarse/conv4-conv/conv4-conv-b/Adam/Initializer/zeros:0
└
'coarse/conv4-conv/conv4-conv-b/Adam_1:0,coarse/conv4-conv/conv4-conv-b/Adam_1/Assign,coarse/conv4-conv/conv4-conv-b/Adam_1/read:029coarse/conv4-conv/conv4-conv-b/Adam_1/Initializer/zeros:0
Ф
coarse/conv4-bn/gamma/Adam:0!coarse/conv4-bn/gamma/Adam/Assign!coarse/conv4-bn/gamma/Adam/read:02.coarse/conv4-bn/gamma/Adam/Initializer/zeros:0
Ь
coarse/conv4-bn/gamma/Adam_1:0#coarse/conv4-bn/gamma/Adam_1/Assign#coarse/conv4-bn/gamma/Adam_1/read:020coarse/conv4-bn/gamma/Adam_1/Initializer/zeros:0
Р
coarse/conv4-bn/beta/Adam:0 coarse/conv4-bn/beta/Adam/Assign coarse/conv4-bn/beta/Adam/read:02-coarse/conv4-bn/beta/Adam/Initializer/zeros:0
Ш
coarse/conv4-bn/beta/Adam_1:0"coarse/conv4-bn/beta/Adam_1/Assign"coarse/conv4-bn/beta/Adam_1/read:02/coarse/conv4-bn/beta/Adam_1/Initializer/zeros:0
╕
%coarse/conv5-conv/conv5-conv-w/Adam:0*coarse/conv5-conv/conv5-conv-w/Adam/Assign*coarse/conv5-conv/conv5-conv-w/Adam/read:027coarse/conv5-conv/conv5-conv-w/Adam/Initializer/zeros:0
└
'coarse/conv5-conv/conv5-conv-w/Adam_1:0,coarse/conv5-conv/conv5-conv-w/Adam_1/Assign,coarse/conv5-conv/conv5-conv-w/Adam_1/read:029coarse/conv5-conv/conv5-conv-w/Adam_1/Initializer/zeros:0
╕
%coarse/conv5-conv/conv5-conv-b/Adam:0*coarse/conv5-conv/conv5-conv-b/Adam/Assign*coarse/conv5-conv/conv5-conv-b/Adam/read:027coarse/conv5-conv/conv5-conv-b/Adam/Initializer/zeros:0
└
'coarse/conv5-conv/conv5-conv-b/Adam_1:0,coarse/conv5-conv/conv5-conv-b/Adam_1/Assign,coarse/conv5-conv/conv5-conv-b/Adam_1/read:029coarse/conv5-conv/conv5-conv-b/Adam_1/Initializer/zeros:0
Ф
coarse/conv5-bn/gamma/Adam:0!coarse/conv5-bn/gamma/Adam/Assign!coarse/conv5-bn/gamma/Adam/read:02.coarse/conv5-bn/gamma/Adam/Initializer/zeros:0
Ь
coarse/conv5-bn/gamma/Adam_1:0#coarse/conv5-bn/gamma/Adam_1/Assign#coarse/conv5-bn/gamma/Adam_1/read:020coarse/conv5-bn/gamma/Adam_1/Initializer/zeros:0
Р
coarse/conv5-bn/beta/Adam:0 coarse/conv5-bn/beta/Adam/Assign coarse/conv5-bn/beta/Adam/read:02-coarse/conv5-bn/beta/Adam/Initializer/zeros:0
Ш
coarse/conv5-bn/beta/Adam_1:0"coarse/conv5-bn/beta/Adam_1/Assign"coarse/conv5-bn/beta/Adam_1/read:02/coarse/conv5-bn/beta/Adam_1/Initializer/zeros:0
А
coarse/fc1/fc1-w/Adam:0coarse/fc1/fc1-w/Adam/Assigncoarse/fc1/fc1-w/Adam/read:02)coarse/fc1/fc1-w/Adam/Initializer/zeros:0
И
coarse/fc1/fc1-w/Adam_1:0coarse/fc1/fc1-w/Adam_1/Assigncoarse/fc1/fc1-w/Adam_1/read:02+coarse/fc1/fc1-w/Adam_1/Initializer/zeros:0
А
coarse/fc1/fc1-b/Adam:0coarse/fc1/fc1-b/Adam/Assigncoarse/fc1/fc1-b/Adam/read:02)coarse/fc1/fc1-b/Adam/Initializer/zeros:0
И
coarse/fc1/fc1-b/Adam_1:0coarse/fc1/fc1-b/Adam_1/Assigncoarse/fc1/fc1-b/Adam_1/read:02+coarse/fc1/fc1-b/Adam_1/Initializer/zeros:0
А
coarse/fc2/fc2-w/Adam:0coarse/fc2/fc2-w/Adam/Assigncoarse/fc2/fc2-w/Adam/read:02)coarse/fc2/fc2-w/Adam/Initializer/zeros:0
И
coarse/fc2/fc2-w/Adam_1:0coarse/fc2/fc2-w/Adam_1/Assigncoarse/fc2/fc2-w/Adam_1/read:02+coarse/fc2/fc2-w/Adam_1/Initializer/zeros:0
А
coarse/fc2/fc2-b/Adam:0coarse/fc2/fc2-b/Adam/Assigncoarse/fc2/fc2-b/Adam/read:02)coarse/fc2/fc2-b/Adam/Initializer/zeros:0
И
coarse/fc2/fc2-b/Adam_1:0coarse/fc2/fc2-b/Adam_1/Assigncoarse/fc2/fc2-b/Adam_1/read:02+coarse/fc2/fc2-b/Adam_1/Initializer/zeros:0"
train_op

Adam"ъ

update_ops█
╪
(coarse/coarse/conv2-bn/AssignMovingAvg:0
*coarse/coarse/conv2-bn/AssignMovingAvg_1:0
(coarse/coarse/conv3-bn/AssignMovingAvg:0
*coarse/coarse/conv3-bn/AssignMovingAvg_1:0
(coarse/coarse/conv4-bn/AssignMovingAvg:0
*coarse/coarse/conv4-bn/AssignMovingAvg_1:0
(coarse/coarse/conv5-bn/AssignMovingAvg:0
*coarse/coarse/conv5-bn/AssignMovingAvg_1:0"
	summaries


loss:0HNвЬ