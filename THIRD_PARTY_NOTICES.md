# Third-party notices

Everything in this repository is MIT-licensed (see [LICENSE](LICENSE)) except the
third-party material below, which keeps its own license.

## Microduck robot model — `32_microduck/robot/`

Copied from [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
at commit `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3` (from `src/mjlab_microduck/robot/microduck/`).

- The MJCF robot descriptions (`.xml`) and configs: Apache License 2.0, Copyright Pollen Robotics.
  See [`32_microduck/robot/LICENSE`](32_microduck/robot/LICENSE).
- The 3D meshes (`32_microduck/robot/assets/*.stl`): Copyright Pollen Robotics, licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial,
  share-alike). See [`32_microduck/robot/LICENSE-MODELS.md`](32_microduck/robot/LICENSE-MODELS.md).
  These meshes are not covered by this repository's MIT license.

## nanochat — `30_nanochat_mlx/`

`30_nanochat_mlx/gpt.py` and `30_nanochat_mlx/optim.py` (and the copies in
`30_nanochat_mlx/efficiency/orig/` and `31_silicon_decathlon/models/nanochat.py`) are MLX ports
of [karpathy/nanochat](https://github.com/karpathy/nanochat), used under the MIT License:

```
MIT License

Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Gymnasium Walker2d model — `19_dance_mechatronics/walker2d_dance.xml`

Based on the `Walker2d-v5` model from [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
used under the MIT License:

```
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
