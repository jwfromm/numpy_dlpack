# Numpy DLPack Array Conversion Example

This example demonstrates how a underlying array memory can be handed off between
two DLPack compatible frameworks without requiring any copies. In this case,
we demonstrate how to convert numpy to TVM's NDArray and vice-versa with proper
memory handling. We hope that not only is this directly useful for TVM users, but
also a solid example for how similar efficient copies can be implemented in other
array frameworks.

## Authors
[Josh Fromm](https://github.com/jwfromm)
[Junru Shao](https://github.com/junrushao1994)
