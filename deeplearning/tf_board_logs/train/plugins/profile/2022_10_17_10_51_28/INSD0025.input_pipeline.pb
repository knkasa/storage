	�KK�?�KK�?!�KK�?	kֻ�8+@kֻ�8+@!kֻ�8+@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�KK�?W^�?���?A��d�VA�?Y�bg
��?rEagerKernelExecute 0*	������Q@2U
Iterator::Model::ParallelMapV2�W����?!R;���<@)�W����?1R;���<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�8*7QK�?!��EU�:@)rk�m�\�?1}�ˌe�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ͩd ��?!y����9@)�-����?1/�f���3@:Preprocessing2F
Iterator::Model����_�?!��IA�C@)`��Ù?1��.�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceY���jq?!��&�@)Y���jq?1��&�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	pz�ǥ?!���N@)�)�TPq?1��a��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3j�J>vg?!�mk�/@)3j�J>vg?1�mk�/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�7��w�?!9��5�;@)�f�v�T?1����S�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 13.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t24.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9kֻ�8+@I3���U@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	W^�?���?W^�?���?!W^�?���?      ��!       "      ��!       *      ��!       2	��d�VA�?��d�VA�?!��d�VA�?:      ��!       B      ��!       J	�bg
��?�bg
��?!�bg
��?R      ��!       Z	�bg
��?�bg
��?!�bg
��?b      ��!       JCPU_ONLYYkֻ�8+@b q3���U@