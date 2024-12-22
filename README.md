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

## System design

For details on DrJAX's system design, check out our
[paper](https://arxiv.org/abs/2403.07128).

## Citation

To cite this repository, please use the following BibTeX citation:

```
@inproceedings{rush2024drjax,
  title={DrJAX: Scalable and Differentiable MapReduce Primitives in JAX},
  author={Rush, J Keith and Charles, Zachary and Garrett, Zachary and Augenstein, Sean and Mitchell, Nicole Elyse},
  booktitle={2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ ICML 2024)}
}
```

## Disclaimers

This is not an officially supported Google product.

If you're interested in learning more about responsible AI practices, please see
Google AI's
[Responsible AI Practices](https://ai.google/education/responsible-ai-practices).

Dataset Grouper is Apache 2.0 licensed. See the [`LICENSE`](LICENSE) file.
