
--------------------------------------------------------------------------------

This is a fork off [PyTorch](https://github.com/pytorch/pytorch) with a `ConditionedLSTMCell` implementation. A `ConditionedLSTMCell` feeds in a fixed context vector, along with the input and state vectors as follows:

--------------------------------------------------------------------------------

*class* **ConditionedLSTMCell** (*input_size, hidden_size, bias=True*):

        \begin{array}{ll}
        i = sigmoid(W_{ii} x + b_{ii} + W_{hi} h + b_{hi} + W_{ci} c + b_{ci}) \\
        f = sigmoid(W_{if} x + b_{if} + W_{hf} h + b_{hf} + W_{cf} c + b_{cf})) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg} + W_{cg} c + b_{cg})) \\
        o = sigmoid(W_{io} x + b_{io} + W_{ho} h + b_{ho} + W_{co} c + b_{co})) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c_t) \\
        \end{array}

    Args:
        input_size: The number of expected features in the input x and context c
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih`, `b_ch' and `b_hh`. Default: True

    Inputs: input, context, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **context** (batch, input_size): tensor containing context features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape `(input_size x hidden_size)`
        weight_ch: the learnable context-hidden weights, of shape `(input_size x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape `(hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_ch: the learnable context-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

Examples::
```
        >>> rnn = nn.ConditionedLSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> context = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], context[i], (hx, cx))
        ...     output.append(hx)
```
