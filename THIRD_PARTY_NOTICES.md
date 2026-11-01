# Third-Party Notices

This repository includes or adapts code from the projects listed below. Source
files that already carry upstream copyright and license headers keep those
headers in place.

## Apache-2.0 Components

The full Apache License 2.0 text is provided in `LICENSE`.

- `scripts/normalize.py` adapts code or structure from
  `Physical-Intelligence/openpi`, licensed under Apache-2.0.
- `wall_x/model/qact/qwen2_5/configuration_qwen2_5_vl.py` and
  `wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl.py` include Qwen /
  Hugging Face derived code, licensed under Apache-2.0.
- The RoboDojo WebSocket transport in `wall_x/serving/robodojo/` adapts the
  `client_server/ws` package of `Luminis-Sim/XPolicyLab`, licensed under
  Apache-2.0. See `wall_x/serving/robodojo/THIRD_PARTY_NOTICES.md`.

## MIT Components

LIBERO evaluation helpers in `wall_x/_vendor/harrix/envs/libero*.py` interface
with LIBERO components, licensed under the MIT License:

Copyright (c) 2023 Lifelong Robot Learning

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
