function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

numSamples = size(data, 2);

%% forward prop
%%% YOUR CODE HERE %%%
% hidden layer: linear transform, Relu/Sigmoid/tanh
for i = 1:numHidden
    if i == 1
        hAct{i} = bsxfun(@plus, stack{i}.W * data, stack{i}.b);
    else
        hAct{i} = bsxfun(@plus, stack{i}.W * hAct{i-1}, stack{i}.b);
    end
    switch ei.activation_fun
        case 'logistic'
            hAct{i} = sigmoid(hAct{i});
        case 'relu'
            hAct{i} = relu(hAct{i});
        case 'tanh'
            hAct{i} = tanh(hAct{i});
       
    end
end

% output layer: softmax
y = exp(bsxfun(@plus, stack{end}.W * hAct{end - 1}, stack{end}.b));
hAct{end} = bsxfun(@rdivide, y, sum(y, 1));
pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
loss = log(hAct{end});
idx = sub2ind(size(loss), labels', 1:numSamples);
ceCost = -sum(loss(idx));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
ground_truth = zeros(size(hAct{end}));
ground_truth(idx) = 1;
grad_prev = hAct{end} - ground_truth;

for i = numHidden + 1:-1:1
    if i == numHidden + 1
        gradLayer = ones(size(grad_prev));
    else
        switch ei.activation_fun
            case 'logistic'
                gradLayer = hAct{i} .* (1 - hAct{i});
            case 'relu'
                gradLayer = hAct{i} > 0;
            case 'tanh'
                gradLayer = 1 - hAct{i} .^ 2;
        end
    end
    gradZ = grad_prev .* gradLayer;
    if i == 1
        gradStack{i}.W = gradZ * data';
    else
        gradStack{i}.W = gradZ * hAct{i - 1}';
    end
    gradStack{i}.b = sum(gradZ, 2);
    grad_prev = stack{i}.W' * gradZ;
    if i~= numHidden + 1
        gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;
    end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

wCost = 0;
for i = 1:numHidden + 1
    wCost = wCost + 0.5 * ei.lambda * sum(stack{i}.W(:) .^ 2);
end
cost = ceCost + wCost;

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



