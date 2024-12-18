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

## Installing

```
pip install --upgrade google-fax
```

## Building a new wheel

Run `python -m build` to build a new `google-fax` wheel.

## Citing FAX

To cite this repository, please use the following BibTeX citation:

```
@misc{rush2024fax,
      title={FAX: Scalable and Differentiable Federated Primitives in JAX},
      author={Keith Rush and Zachary Charles and Zachary Garrett},
      year={2024},
      eprint={2403.07128},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

## Disclaimers

This is not an officially supported Google product.

If you're interested in learning more about responsible AI practices, please see
Google AI's
[Responsible AI Practices](https://ai.google/education/responsible-ai-practices).

Dataset Grouper is Apache 2.0 licensed. See the [`LICENSE`](LICENSE) file.
