# DrJAX - Differentiable MapReduce Primitives in JAX

DrJAX is a library designed to embed a MapReduce programming model into JAX. DrJAX has
multiple objectives.

1.  Create a simple JAX-based authoring surface for MapReduce computations.
1.  Leverage JAX's sharding mechanisms to enable highly optimized execution of
    MapReduce computations, especially in large-scale datacenter settings.
1.  Full differentiability of DrJAX computations, including differentiating
    through communication primitives like broadcasts and reductions.

DrJAX is designed to make it easy to author and execute parallel computations in
the datacenter. DrJAX is tailored towards **large-scale** parallel and distributed computations,
including computations involving larger models, and ensuring that they can be
run efficiently. DrJAX embeds primitives like those defined by
[TensorFlow Federated](https://github.com/tensorflow/federated) using the
mapping capabilities and primitive extensions of JAX.
