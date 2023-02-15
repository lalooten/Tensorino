function C = contract(tensors, indices, kwargs)
% Compute a tensor network contraction.
%
% Usage
% -----
% :code:`C = contract(t1, idx1, t2, idx2, ...)`
%
% :code:`C = contract(..., 'Conj', conjlist)`
%
% Repeating Arguments
% -------------------
% tensors : :class:`Tensor`
%   list of tensors that constitute the vertices of the network.
%
% indices : int
%   list of indices that define the links and contraction order, using ncon-like syntax.
%
% Keyword Arguments
% -----------------
% Conj : (1, :) :class:`logical`
%   optional list to flag that tensors should be conjugated.
%
% Sequence : (1, :) :class:`int`
%   optional custom contraction sequence.
%
% Returns
% -------
% C : :class:`Tensor` or numeric
%   result of the tensor network contraction.

% TODO contraction order checker.

arguments (Repeating)
    tensors
    indices (1, :) {mustBeInteger}
end

arguments
    kwargs.Sequence (1, :) = 1:max(cellfun(@max, indices))
end

for i = 1:length(tensors)
    if length(indices{i}) > 1
        assert(length(unique(indices{i})) == length(indices{i}), ...
            'Tensors:TBA', 'Traces not implemented.');
    end
end

contractindices = cellfun(@(x) x(x > 0), indices, 'UniformOutput', false);

% Reorder contraction sequence
sequence = kwargs.Sequence;
if ~issorted(sequence)
    invseq(sequence) = 1:length(sequence);
    contractindices = cellfun(@(x) invseq(x), contractindices, 'UniformOutput', false);
end
    
% Special case for single input tensor
if nargin == 2
    [~, order] = sort(indices{1}, 'descend');
    C = tensors{1};
    C = permute(C, order);
    return
end

% Generate trees
partialtrees = num2cell(1:length(tensors));
tree = generatetree(partialtrees, contractindices);

% contract all subtrees
[A, ia] = contracttree(tensors, indices, tree{1});
[B, ib] = contracttree(tensors, indices, tree{2});

% contract last pair
[dimA, dimB] = contractinds(ia, ib);

C = tensorprod(A, B, dimA, dimB, 'NumDimensionsA', length(ia));
ia(dimA) = [];  ib(dimB) = [];
ic = [ia ib];

% permute last tensor
if ~isempty(ic) && length(ic) > 1
    [~, order] = sort(ic, 'descend');
    C = permute(C, order);
end

end

function [dimA, dimB] = contractinds(ia, ib)
% contractinds - Find the contracted dimensions.
%   [dimA, dimB] = contractinds(ia, ib)
%       locates the repeated indices.

ind = find(ia(:) == ib).' - 1;
dimA = mod(ind, length(ia)) + 1;
dimB = floor(ind / length(ia)) + 1;

end

function tree = generatetree(partialtrees, contractindices)

if length(partialtrees) == 1
    tree = partialtrees{1};
    return
end

if all(cellfun('isempty', contractindices)) % disconnected network
    partialtrees{end - 1} = partialtrees(end - 1:end);
    partialtrees(end) = [];
    contractindices(end) = [];
else
    tocontract = min(horzcat(contractindices{:}));
    tinds = find(cellfun(@(x) any(tocontract == x), contractindices));
    
    if length(tinds) ~= 2
        celldisp(contractindices);
        error('contract:indices', ...
            'contracted indices should appear exactly twice.\n(%d)', ...
            tocontract);
    end
    partialtrees{tinds(1)} = partialtrees(tinds);
    partialtrees(tinds(2)) = [];
    contractindices{tinds(1)} = unique1(horzcat(contractindices{tinds}));
    contractindices(tinds(2)) = [];
end

tree = generatetree(partialtrees, contractindices);

end

function [C, ic] = contracttree(tensors, indices, tree)

if isnumeric(tree)
    C = tensors{tree};
    ic = indices{tree};
    return
end

[A, ia] = contracttree(tensors, indices, tree{1});
[B, ib] = contracttree(tensors, indices, tree{2});
[dimA, dimB] = contractinds(ia, ib);

C = tensorprod(A, B, dimA, dimB, 'NumDimensionsA', length(ia));

ia(dimA) = [];
ib(dimB) = [];
ic = [ia ib];

end

function inds = unique1(inds)

inds = inds(sum(inds(:) == inds) == 1);

end

