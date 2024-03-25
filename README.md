## mamba-minimal

Simple, minimal implementation of Mamba in one file of PyTorch.

Featuring:
* Equivalent numerical output as official implementation for both forward and backward pass
* Simplified, readable, annotated code

Does NOT include:
* Speed. The official implementation is heavily optimized, and these optimizations are core contributions of the Mamba paper. I kept most implementations simple for readability.
* Proper parameter initialization (though this could be added without sacrificing readability)

## Demo

See [demo.ipynb](demo.ipynb) for examples of prompt completions.

- The original version
```python
from model.MixerModel import MixerModel
from transformers import AutoTokenizer

model = MixerModel.from_pretrained(('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba is the only other game in the series that I haven't played before. My favourite of the series, in fact. Its been very well received by gamers and I look forward to getting my hands on it when it gets released.
>
>Well, this is


- My version

```python
from model.MixerModel import MixerModel
from transformers import AutoTokenizer

model = MixerModel.from_pretrained(('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate2(model, tokenizer, 'Mamba is the')
```
> Mamba is the only animal killed by a single bite from a single Mamba. Most of the bites are from the Asian Boa Constrictor. They are very rare and can vary from very mild to very severe. I found two bites to be so painful. One

- The time improve\
generate:   24.2s \
generate2:  5.3s

This change utilizes the `InferenceParameter` (following the practices of the original author) to save time by avoiding re-reading words that have already been processed by Mamba.

## References

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
