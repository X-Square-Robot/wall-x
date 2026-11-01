# Third-Party Notices

The RoboDojo-compatible WebSocket transport in this directory is adapted from
the `client_server/ws` package in
[Luminis-Sim/XPolicyLab](https://github.com/Luminis-Sim/XPolicyLab), revision
`ef2cc0de8604a901b9491cf166bb9e1f26d72e0b`, together with local compatibility
changes used by Wall-X.

The adapted files are distributed under the Apache License, Version 2.0. The
repository-level `LICENSE` file contains the full license text. The code has
been modified to use the `wall_x.serving.robodojo` namespace and includes
Wall-X-specific JPEG ndarray, dynamic batch, request error, and timing support.
