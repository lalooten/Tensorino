classdef (InferiorClasses = {?sym}) Tensorino
    % Class for multi-dimensional sparse tensors.
    %
    %   Handles sparse tensors with an unlimited number of elements, with the restriction
    %   that each individual index dimension must be smaller than 2^32-1.
    
    %% Properties
    properties (Access = private)
        ind = [] % (:, :) double
        var = [] % (:, 1) double
        sz = [] % (1, :) double
    end
    
    %% Constructors
    methods
        
        function t = Tensorino(varargin)
            % Create a Tensorino.
            %
            % Usage
            % -----
            % :code:`t = Tensorino(ind, vals, sz)`
            %   Uses the rows of :code:`ind` and :code:`vals` to generate a
            %   :class:`Tensorino` :code:`t` of size :code:`sz = [m1 m2 ... mn]`.
            %   :code:`ind` is a :code:`p` x :code:`n` array specifying the subscripts of
            %   the nonzero values to be inserted into :code:`t`. The k-th row of
            %   :code:`ind` specifies the subscripts for the k-th value in :code:`vals`.
            %   The argument :code:`vals` may be scalar, in which case it is expanded to be
            %   the same length as :code:`ind`, i.e., it is equivalent to
            %   :code:`vals * (p, 1)`.
            %
            % :code:`t = Tensorino(ind, vals)`
            %   Same as the three-argument call, where now the size of the
            %   :class:`Tensorino` is taken to be the column-wise maximum of :code:`ind`.
            %
            % :code:`t = Tensorino`
            %   Empty constructor.
            %
            % :code:`t = Tensorino(a)`
            %   Copies/converts :code:`a` if it is a :class:`Tensorino`, a dense array or a
            %   sparse matrix.
            %
            % :code:`t = Tensorino(a, sz)`
            %   Copies/converts :code:`a` if it is a :class:`Tensorino`, a dense array or a
            %   sparse matrix, and sets the size of :code:`a` to :code:`sz`
            % 
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> ind = [1 1 1; 1 1 3; 2 2 2; 4 4 4];
            %   >> vals = [0.5; 1.5; 2.5; 3.5];
            %   >> sz = [4 4 4];
            %   >> t = Tensorino(ind, vals, sz) %<-- 4x4x4 Tensorino
            %
            % Note
            % ----
            % Currently the :class:`Tensorino` constructor does not allow duplicate
            % subscripts in :code:`ind`, unlike the builtin Matlab :code:`sparse`.
            
            if (nargin == 0) || ((nargin == 1) && isempty(varargin{1}))
                % Empty constructor
                return;

            elseif nargin == 1
                % Single argument: copy or cast

                source = varargin{1};

                switch(class(source))

                    % copy constructor
                    case 'Tensorino'
                        t.ind = source.ind;
                        t.var = source.var;
                        t.sz = source.sz;
                                                
                    % sparse matrix, dense array; limited in size
                    case {'numeric', 'logical', 'double', 'sym'}
                        nz = find(source);
                        t.ind = ind2sub_(size(source), nz);
                        t.var = squeeze(reshape(source(nz), [], 1)); % TODO: why the squeeze?
                        if issparse(source) % TODO: necessary?
                            t.var = full(t.var);
                        end
                        t.sz = size(source);
                    otherwise
                        error('Tensorino:UnsupportedConstructor', 'Unsupported use of Tensorino constructor.');

                end
                
            elseif nargin == 2
                % Two arguments: indices, values
                ind = varargin{1};
                var = varargin{2}(:); % TODO: if reshape is too expensive, can put error check
                if isscalar(var)
                    var = repmat(var, size(ind, 1), 1);
                end
                mask = var ~= 0;
                t.ind = ind(mask);
                t.var = var(mask);
                t.sz = max(t.ind, [], 1);
                
            elseif nargin == 3
                % Three arguments: indices, values, size
                ind = varargin{1};
                var = varargin{2}(:);
                if isscalar(var)
                    var = repmat(var, size(ind, 1), 1);
                end
                sz = varargin{3}(:).';
                mask = var ~= 0;
                ind = ind(mask, :);
                var = var(mask);
%                 % add values for duplicate subscripts
%                 [ind, ia, ic] = unique(ind, 'rows');
%                 % no way to do this without looping it seems...
                % assign properties
                t.ind = ind;
                t.var = var;
                t.sz = sz;
            else
                error('Tensorino:UnsupportedConstructor', 'Unsupported use of Tensorino constructor.');
            end
            
            % Error checking; TODO: do we want this?
            if size(t.ind, 1) ~= size(t.var, 1)
                error('Tensorino:InvalidProperties', 'Indices and values must be same size');
            end
            if ~isempty(t.ind)
                if size(t.ind, 2) ~= size(t.sz, 2)
                    error('Tensorino:InvalidProperties', 'Number of indices does not match size vector');
                end
                if any(max(t.ind, [], 1) > t.sz)
                    error('Tensorino:InvalidProperties', 'Indices exceed size vector');
                end
            end
        end
        
    end
    
    %% Methods
    methods

        function t = abs(t)
            % Absolute value and complex magnitude.
            %
            % Usage
            % -----
            % :code:`b = abs(t)`
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   Input tensor.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   Absolute value of each element in :code:`t`.
            %   If :code:`t` is complex, :code:`abs(t)` returns the complex magnitude.
            t.var = abs(t.var);
        end
        
        function t_res = app(t, mat, ind)
            % Apply rescaling matrices on certain indices.
            %
            % Usage
            % -----
            % :code:`t_res = app(t, mat1, ind1, mat, ind2, ...)`
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   Input tensor.
            %
            % Repeating Arguments
            % -------------------
            % mat : :class:`Tensorino`
            %   Rescaling matrices to be multiplied with :code:`t`.
            %
            % ind : :class:`int`
            %   Indices onto which rescaling matrices are applied.
            %
            % Returns
            % -------
            % t_res : :class:`Tensorino`
            %   Rescaled tensor.
            arguments
                t
            end
            arguments (Repeating)
                mat
                ind
            end
            con = cell(1, numel(ind) + 1);
            ten = cell(1, numel(ind) + 1);
            ten{1} = t;
            con{1} = -1:-1:-ndims(t);
            con{1}([ind{:}]) = 1:numel(ind);
            for i = 1:numel(ind)
                con{1+i} = [i -ind{i}];
                ten{1+i} = mat{i};
            end
            conargs = reshape([ten; con], 1, []);
            t_res = contract(conargs{:});
        end
        
        function t = conj(t)
            % Complex conjugate.
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array.
            t.var = conj(t.var);
        end
        
        function a = ctranspose(a)
            % Complex conjugate transpose.
            %
            % Only defined for 2D :class:`Tensorino` s.
            %
            % Usage
            % -----
            % :code:`b = ctranspose(a)`
            %
            % :code:`b = a'`
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array.
            assert(ismatrix(a), 'Tensorino:RankError', 'ctranspose is only defined for 2D Tensorino.');
            a = permute(conj(a), [2, 1]);
        end
        
        function disp(t)
            nz = nnz(t);

            if nz == 0
                fprintf('all-zero %s of size %s\n\n', class(t), ...
                    dim2str(t.sz));
                return
            end

            fprintf('%s of size %s with %d nonzeros:\n\n', class(t), ...
                dim2str(t.sz), nz);

            if (nz > 1000)
                r = input('Big Tensorino, do you want to print all nonzeros? (y/n) ', 's');
                if ~strcmpi(r, 'y'), return, end
            end

            spc = floor(log10(max(double(t.ind), [], 1))) + 1;
            fmt_ind = sprintf('%%%du,', spc(1:end-1));
            fmt_ind = sprintf('%s%%%du', fmt_ind, spc(end));
            fmt = sprintf('\t(%s)%%s\n', fmt_ind);
            S = evalc('disp(t.var)'); % abuse builtin display
            S = splitlines(S);
            S = S(~cellfun(@isempty, S));
            if contains(S{1}, '*') % big numbers
                fprintf('%s\n\n', S{1});
                S = S(2:end);
            end
            for i = 1:nz
                fprintf(fmt, t.ind(i, :), S{i});
            end
            fprintf('\n')
        end
        
        function d = distance(t1, t2)
            % Compute the Euclidean distance between two :class:`Tensorino`s.
            %
            % Arguments
            % ---------
            % t1, t2 : :class:`Tensorino`
            %
            % Returns
            % -------
            % d : :class:`double`
            %   Euclidean distance, defined as the norm of the difference.
                        
            d = norm(t1 - t2);
        end
        
        function d = dot(t1, t2)
            % Compute the scalar dot product of two :class:`Tensorino`s. This is defined as
            % the overlap of the two tensors, which therefore must have sizes. This
            % function is sesquilinear in its arguments.
            %
            % Arguments
            % ---------
            % t1, t2 : :class:`Tensorino`
            %   Tensors of equal structure.
            %
            % Returns
            % -------
            % d : :class:`double`
            %   Scalar dot product of the two tensors.
            assert(isequal(size(t1), size(t2)), 'Tensorino:dimerror', ...
                'input tensors must have the same size.');
            d = contract(conj(t1), 1:ndims(t1), t2, 1:ndims(t2));
        end
        
        function [V, D] = eig(t)
            % Find eigenvalues and eigenvectors of a square :class:`Tensorino`.
            %
            % Usage
            % -----
            % :code:`e = eig(t)`
            %
            % :code:`[V, D] = eig(t)`
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   Input tensor of size :code:`[a b ... z a b ... z]`, interpreted as a
            %   (:code:`a` x :code:`a` x ... x :code:`z`) x (:code:`a` x :code:`a` x ... x
            %   :code:`z`) matrix.
            %
            % Returns
            % -------
            % e : :class:`double`
            %   Column vector containing the eigenvalues of :code:`t`.
            %
            % D : (:, :) :class:`Tensorino`
            %   Diagonal :class:`Tensorino` of eigenvalues.
            %
            % V : (1, :) :class:`Tensorino`
            %   Row vector of right eigenvectors such that :code:`A * V = V * D`.
            % 
            % Todo
            % ----
            % Decide on/document index splitting convention (fixing domain and codomain for
            % matrix decomposition).
            %
            % See Also
            % --------
            % `Documentation for builtin Matlab eig <https://mathworks.com/help/matlab/ref/eig.html>`_.
            nd = ndims(t);
            if mod(nd, 2) || any(t.size(1:nd/2) ~= t.size(nd/2+1:end))
                error("eig only works for (a x b x ... x z) x (a x b x ... x z) Tensorinos")
            end
            if isempty(t.var)
                V = [];
                D = [];
                return;
            end

            [U, ~, IC] = unique([t.ind(:, 1:nd/2); t.ind(:, nd/2+1:end)], 'rows');
            r = IC(1:length(t.ind(:, 1)));
            c = IC(length(t.ind(:, 1)) + 1:end);
            sz = max([r; c]);
            m = full(sparse(r, c, t.var, sz, sz));
            fprintf("Calculating eigenvalues of %i x %i matrix ...\n", sz, sz);

            if nargout <= 1
                V = eig(m);
            else
                [V, D] = eig(m);
                [vi, vj, vv] = find(V);
                V = Tensorino([U(vi, :) vj], vv, [t.size(1:nd/2) size(D, 1)]);
            end
        end
        
        function [V, D] = eigs(t, n)
            % Find a few eigenvalues and eigenvectors of a :class:`Tensorino`.
            %
            % Todo
            % ----
            % Decide on/document index splitting convention (fixing domain and codomain for
            % matrix decomposition).
            %
            % See Also
            % --------
            % `Documentation for builtin Matlab eigs <https://mathworks.com/help/matlab/ref/eigs.html>`_.
            nd = ndims(t);
            if mod(nd, 2) || all(t.size(1:nd/2) ~= t.size(nd/2+1:end))
                error("eigs only works for (a x b x ... x z) x (a x b x ... x z) Tensorinos")
            end
            if isempty(t.var)
                V = [];
                D = [];
                return;
            end

            [U, ~, IC] = unique([t.ind(:, 1:nd/2); t.ind(:, nd/2+1:end)], 'rows');
            r = IC(1:length(t.ind(:, 1)));
            c = IC(length(t.ind(:, 1)) + 1:end);
            sz = max([r; c]);
            m = sparse(r, c, t.var, sz, sz);
            fprintf("Calculating eigenvalues of %i x %i matrix ...\n", sz, sz);

            if nargout <= 1
                if nargin == 1
                    V = eigs(m);
                else
                    V = eigs(m, n);
                end
            else
                if nargin == 1
                    [V, D] = eigs(m);
                else
                    [V, D] = eigs(m, n);
                end
                [vi,vj,vv] = find(V);
                V = Tensorino([U(vi, :) vj], vv, [t.size(1:nd/2) size(D, 1)]);
            end
        end
        
        function export(t, digits)
            dlmwrite(inputname(1) + "_ind.txt", t.ind);
            dlmwrite(inputname(1) + "_var.txt", t.var, 'precision', digits);
            dlmwrite(inputname(1) + "_size.txt", t.size);
        end
        
        function [ind, var] = find(t)
            % Find subscripts of nonzero elements in a :class:`Tensorino`.
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array
            %
            % Returns
            % -------
            % ind : (:, :) :class:`int`
            %   subscripts of nonzero entries.
            %
            % var : (:, 1) :class:`double`
            %   values of nonzero entries.
            ind = t.ind;
            var = t.var;
        end
        
        function t_full = full(t)
            % Convert a :class:`Tensorino` to a dense array.
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % d : :class:`double`
            %   dense output array.
            %
            % Warning
            % -------
            % Limited in size!
            if isa(t.var, 'sym')
                t_full = sym(zeros(numel(t), 1)); % TODO: call appropriate zeros through t.var.zeros(...)
            else
                t_full = zeros(numel(t), 1);
            end
            t_full(sub2ind_(t.size, t.ind)) = t.var;
            if ndims(t) > 1
                t_full = reshape(t_full, t.size);
            end
        end
        
        function t = groupind(t, g)
            % Group (block) specified sets of contiguous indices.
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % g : (1, :) :class:`int`
            %   list of number of contiguous indices to be grouped in each index of the
            %   output tensor.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output tensor with grouped indices.
            %
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> t = Tensorino.random([2, 3, 4, 5, 6], .1);
            %   >> b = groupind(t, [3, 2]); %<-- 24x30 Tensorino
            assert(sum(g) == ndims(t), 'Tensorino:InvalidGrouping', 'Invalid grouping.')
            r = zeros(1, length(g));
            offset = 0;
            for i = 1:length(g)
                r(i) = prod(t.size(1+offset:g(i)+offset));
                offset = offset + g(i);
            end
            t = reshape(t, r);
        end
        
        function t = imag(t)
            % Complex imaginary part of sparse tensor.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array with real entries corresponding to the imaginary part of the
            %   entries of :code:`a`.
            t.var = imag(t.var);
        end
        
        function bool = isequal(t1, t2)
            % Compare equality of two :class:`Tensorino`s.
            %
            % Arguments
            % ---------
            % t1, t2 : :class:`Tensorino`
            %   input tensors to be compared.
            %
            % Returns
            % -------
            % bool : :class:`logical`
            %   comparison result.
            [ind1, s1] = sortrows(t1.ind);
            [ind2, s2] = sortrows(t2.ind);
            v1 = t1.var;
            v2 = t2.var;
            bool = true;
            if ~isequal(ind1, ind2) || ~isequal(v1(s1), v2(s2))
                bool = false;
            end
        end
        
        function bool = isnumeric(~)
            % Determine whether input is numeric.
            % 
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input tensor.
            %
            % Returns
            % -------
            % bool : :class:`logical`
            %   defaults to :code:`true` for :class:`Tensorino`.
            bool = true;
        end
        
        function bool = isscalar(~)
            % Determine whether input is scalar.
            % 
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input tensor.
            %
            % Returns
            % -------
            % bool : :class:`logical`
            %   defaults to :code:`false` for :class:`Tensorino`.
            bool = false;
        end
        
        function bool = isrow(t)
            % Determine whether input is row vector.
            % 
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input tensor.
            %
            % Returns
            % -------
            % bool : :class:`logical`
            bool = ismatrix(t) && size(t, 1) == 1;
        end
        
        function bool = isstruct(~)
            bool = false;
        end

        function bool = ismatrix(t)
            bool = ndims(t) == 2;
        end
        
        function c = ldivide(a, b)
            % Elementwise left division for :class:`Tensorino`.
            %
            % Usage
            % -----
            % 
            % :code:`ldivide(a, b)`
            %
            % :code:`a .\ b`
            %
            % Arguments
            % ---------
            % a : :class:`double`
            %   Scalar to divide by
            %
            % b : :class:`Tensorino`
            %   Input tensor.
            %
            % Returns
            % -------
            % c : :class:`Tensorino`
            %   Output tensor.
            %
            % Note
            % ----
            % Currently restricted to the case where :code:`a` is a scalar.
            c = rdivide(b, a);
        end
        
        function c = minus(t1, t2)
            % Elementwise subtraction for :class:`Tensorino`s.
            %
            % Note
            % ----
            % Scalars are only subtracted from nonzero entries of the :class:`Tensorino`.
            %
            % Usage
            % -----
            % 
            % :code:`minus(a, b)`
            %
            % :code:`a - b`
            %
            % :code:`a` and :code:`b` must have the same size,
            % unless one is a scalar. A scalar can be subtracted from a :class:`Tensorino`
            % of any size.   
            %
            % Arguments
            % ---------
            % t1, t2 : :class:`Tensorino` or :class:`double`
            %   input tensors.
            %
            % Returns
            % -------
            % t : :class:`Tensorino`
            %   output tensor.
            c = plus(t1, -t2);
        end
        
        function a = mrdivide(a, b)
            % Matrix right division for :class:`Tensorino`.
            %
            % Usage
            % -----
            % :code:`mrdivide(a, b)`
            %
            % :code:`a / b`
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   Input tensor.
            %
            % b : :class:`double`
            %   Scalar to divide by
            %
            % Returns
            % -------
            % c : :class:`Tensorino`
            %   Output tensor.
            %
            % Note
            % ----
            % Currently restricted to the case where :code:`b` is a scalar.
            a = a ./ b;
        end
                
        function t = mtimes(a, b)
            % Matrix multiplication for :class:`Tensorino`s.
            %
            % Usage
            % -----
            % 
            % :code:`t = mtimes(a, b)`
            %
            % :code:`t = a * b`
            %
            % Note
            % ----
            % Currently restricted to the case where either :code:`a` or :code:`b` is a
            % scalar.
            t = a .* b;
        end
        
        function n = ndims(t)
            % Number of dimensions of a :class:`Tensorino`.
            %
            % Note that unlike the `builtin Matlab behavior
            % <https://mathworks.com/help/matlab/ref/double.ndims.html>`_
            % trailing singleton dimensions are not ignored.
            %
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> a = Tensorino.random([4 3 1], .1);
            %   >> ndims(a) %<-- returns 3
            n = size(t.sz, 2);
        end
        
        function n = nnz(t)
            % Number of nonzero elements in a :class:`Tensorino`.
            n = size(t.var, 1);
        end
                
        function nrm = norm(t)
            % Frobenius norm of a :class:`Tensorino`.
            nrm = norm(t.var);
        end
        
        function t = normalize(t)
            % Normalize :class:`Tensorino`.
            t = t ./ norm(t);
        end
        
        function n = numArgumentsFromSubscript(varargin)
            n = 1;
        end
        
        function n = numel(t)
            % Number of elements in a sparse array.
            n = prod(size(t));
        end
        
        function c = outer(a, b)
            % Outer product of two :class:`Tensorino`s.
            %
            % Warning
            % -------
            % Not :code:`kron`.
            if isa(a, 'Tensorino')
              [indA, varA] = find(a);
            else
                indA = ind2sub_(size(a), find(a));
                varA = nonzeros(a);
            end
            if isa(a, 'Tensorino')
                [indB, varB] = find(b);
            else
                indB = ind2sub_(size(b), find(b));
                varB = nonzeros(b);
            end
            varC = kron(varA, varB);
            indC = [kron(indA, ones(length(varB),1)), kron(ones(length(varA), 1), indB)];
            sizeC = [size(a), size(b)];
            c = Tensorino(indC, varC, sizeC);
        end
        
        function t = permute(t, order)
            % Permute :class:`Tensorino` dimensions.
            if ~isempty(t.ind)
                t.ind = t.ind(:, order);
            end
            t.sz = t.sz(order);
        end
                
        function t = plus(t1, t2)
            % Elementwise addition for :class:`Tensorino`s. 
            %
            % Note
            % ----
            % Scalars are only added to nonzero entries of the :class:`Tensorino`.
            %
            % Usage
            % -----
            % 
            % :code:`plus(a, b)`
            %
            % :code:`a + b`
            %
            % :code:`a` and :code:`b` must have the same size,
            % unless one is a scalar. A scalar can be added to a sparse array of any size.
            %
            % Arguments
            % ---------
            % t1, t2 : :class:`Tensorino` or :class:`double`
            %   input tensors.
            %
            % Returns
            % -------
            % t : :class:`Tensorino`
            %   output tensor
            if isa(t2, 'double')
                t = Tensorino(t1.ind, t1.var + t2, t1.size);
                return;
            elseif isa(t1, 'double')
                t = Tensorino(t2.ind, t2.var + t1, t2.size);
                return;
            end
            assert(isequal(size(t1), size(t2)), 'Dimensions must agree');
            if isempty(t1.var)
                t = t2;
                return;
            end
            if isempty(t2.var)
                t = t1;
                return;
            end

            [common, i1, i2] = intersect(t1.ind, t2.ind, 'rows');
            p1 = true(size(t1.ind, 1), 1);
            p2 = true(size(t2.ind, 1), 1);
            p1(i1) = false;
            p2(i2) = false;
            ind = [t1.ind(p1, :); t2.ind(p2, :); common];
            var = [t1.var(p1); t2.var(p2); t1.var(i1) + t2.var(i2)];
            t = Tensorino(ind, var, t1.size);
        end
        
        function a = power(a, b)
            % Elementwise power for :class:`Tensorino`.
            assert(isscalar(b), 'Tensorino:NonScalarPower', 'Tensorino only supports power with a scalar.')
            a.var = a.var.^b;
        end

        function t = purge(t, tol)
            % Set all nonzero values in :class:`Tensorino` who's absolute value is below
            % a given threshold to zero.
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % tol : :class:`float` , optional
            %   threshold tolerance for absolute values of entries, defaults to
            %   :code:`1e-15`.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   sparse array with entries of absolute value below :code:`tol` set to zero.
            arguments
                t
                tol = 1e-15
            end
            
            prg = abs(t.var) < tol;
            t.var(prg) = [];
            t.ind(prg, :) = [];
        end
        
        function a = rdivide(a, b)
            % Elementwise right division for :class:`Tensorino`.
            %
            % Usage
            % -----
            % :code:`rdivide(a, b)`
            %
            % :code:`a ./ b`
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   Input tensor.
            %
            % b : :class:`double`
            %   Scalar to divide by
            %
            % Returns
            % -------
            % c : :class:`Tensorino`
            %   Output array.
            %
            % Note
            % ----
            % Currently restricted to the case where :code:`b` is a scalar.
            assert(isscalar(b), 'Tensorino:ScalarDivide', 'Tensorino.rdivide only supports the scalar case.')
            a.var = a.var ./ b;
        end

        
        function t = real(t)
            % Complex real part of a :class:`Tensorino`.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array with real entries corresponding to the real part of the
            %   entries of :code:`a`.
            t.var = real(t.var);
        end
        
        function [t,uniq_ind] = reduce(t, dims)
            % Reduce the size of a :class:`Tensorino` along given dimensions based on the
            % subscripts of its nonzero elements by dropping all unused index values for
            % each given dimension individually.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input array.
            %
            % dims : (1, :) :class:`int`, optional
            %   dimensions to be reduced, defaults to :code:`ind = 1:ndims(t)`.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array.
            arguments
                t
                dims = 1:ndims(t)
            end
            s = size(t);
            uniq_ind = cell(1, length(t.size));
            for i = dims
                [uniq_ind{i}, ~, t.ind(:, i)] = unique(t.ind(:, i));
                s(i) = max(t.ind(:, i), [], 1);
            end
            t.sz = s;
        end
                
        function t = rescale(t, dims, factors)
            % Apply rescaling factors along given :class:`Tensorino` dimensions.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input array.
            %
            % dims : (1, :) :class:`int`
            %   dimensions to be rescaled.
            %
            % factors : (1, :) :class:`cell`
            %   cell array of rescaling factors for each dimension, where :code:`factors{i}`
            %   is a vector of size :code:`size(t, dims(i))`.
            %
            % Returns
            % -------
            % t_res : :class:`Tensorino`
            %   rescaled tensor.
            assert(length(dims) == length(factors), "Number of rescaled indices must match number of rescaling factors");
            for i = 1:length(dims)
                d = dims(i);
                assert(size(t, d) == length(factors{i}), sprintf("Dimension mismatch on index %i", d));
                t.var = t.var .* reshape(factors{i}(t.ind(:, d)), [], 1);
            end
        end
        
        function t_new = reshape(t, newsize)
            % Reshape a :class:`Tensorino`.
            oldsize = t.sz;
            assert(prod(newsize) == prod(oldsize), "To reshape the number of elements must not change.")
            assert(~any(newsize > 2^32-1), "Individual Tensorino dimensions must not exceed MAXSIZE.")
            newind = sub2sub_(newsize, oldsize, t.ind);
            t_new = Tensorino(newind, t.var, newsize);
        end
        
        function t = simplify(t)
            % Simplify a symbolic :class:`Tensorino`.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input symbolic tensor.
            %
            % Returns
            % -------
            % t_sim : :class:`Tensorino`
            %   simplified symbolic tensor.
            t.var = simplify(t.var);
        end

        function d = size(t, dim)
            % :class:`Tensorino` dimensions.
            % 
            % Usage
            % -----
            % :code:`d = size(a)`
            %   returns the size of the Tensorino.
            %
            % :code:`d = size(a, dim)`
            %   returns the sizes of the dimensions specified by :code:`dim`, which is
            %   either an integer or a vector of integer dimensions.
            if nargin > 1
                d = t.sz(dim);
            else
                d = t.sz;
            end
        end
        
        function [m, Ur, Uc] = sparse(t, g_ind)
            % Convert :class:`Tensorino` to a sparse matrix.
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input  tensor.
            %
            % g_ind : (1, 2) :class:`int`, optional
            %   list of number of contiguous indices to be grouped in each index of the
            %   output sparse matrix. If :code:`ndims(t) ~= 2`, then :code:`g_ind` must be
            %   provided.
            %
            % Returns
            % -------
            % m : :class:`sparse`
            %   output sparse matrix, possibly with grouped indices.
            %
            % Ur : ???
            %
            % Uc : ???
            %
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> ???
            s = length(t.size);
            if nargin == 1
                assert(s == 2, 'Grouped indices not provided');
                if isempty(t.var)
                    m = sparse([], [], [], size(t, 1), size(t, 2));
                else
                    m = sparse(double(t.ind(:,1)),double(t.ind(:,2)),t.var,t.size(1),t.size(2));
                end
            elseif nargin == 2
                assert(length(g_ind) == 2 && sum(g_ind) == s, 'Tensorino:InvalidGrouping', 'Invalid grouping.')
                if isequal(size(t, 1:g_ind(1)), size(t, g_ind(1)+1:g_ind(2))) % square output matrix
                    [Ur, ~, IC] = unique([t.ind(:, 1:g_ind(1)); t.ind(:, g_ind(1)+1:g_ind(1)+g_ind(2))], 'rows');
                    r = IC(1:length(t.ind(:, 1)));
                    c = IC(length(t.ind(:, 1))+1:end);
                    sz1 = max([r; c]);
                    sz2 = sz1;
                else
                    [Ur, ~, r] = unique(t.ind(:, 1:g_ind(1)), 'rows');
                    [Uc, ~, c] = unique(t.ind(:, g_ind(1)+1:end), 'rows');
                    sz1 = max(r);
                    sz2 = max(c);
                end
                m = sparse(r, c, t.var,  sz1, sz2);
            else
                error('Unsupported use of sparse');
            end
        end
        
        function t = squeeze(t)
            % Remove singleton dimensions from a :class:`Tensorino`.
            %
            % Usage
            % -----
            % :code:`b = squeeze(a)` 
            %   returns a sparse tensor :code:`b` with the same elements as :code:`a` but
            %   with all the singleton dimensions removed.
            %
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> squeeze(Tensorino.random([2, 1, 3], 0.5)) %<-- returns a 2 x 3 Tensorino
            %   >> squeeze(Tensorino([1, 1, 1], 1, [1, 1, 1])) %<-- returns a scalar
            if sum(t.sz > 1) == 0
                t = full(t.var);
                return
            end
            t.ind = t.ind(:, size(t) > 1);
            t.sz = t.sz(size(t) > 1);
        end
        
        function t = subs(t,symbols,values)
            % Symbolic substitution for a :class:`Tensorino`.
            %
            % See Also
            % --------
            % Builtin `subs <https://mathworks.com/help/symbolic/subs.html>`_.
            t.var = subs(t.var, symbols, values);
        end
        
        function t = subsasgn(t, s, b)
            % Subscripted assignment for a :class:`Tensorino`.
            %
            % Todo
            % ----
            % Write docstring.
            switch s(1).type
                case '()'
                    subs = cell2mat(s.subs')';
                    if size(subs, 1) ~= 1
                        error('Only elementwise assignment is supported at the moment');
                    end
                    if ~isempty(t.ind)
                        idx = find(all(t.ind == subs, 2));
                        if isempty(idx)
                            t.ind = [t.ind; subs];
                            t.var = [t.var; b];
                        else
                            t.ind(idx, :) = subs;
                            t.var(idx) = b;
                        end
                    else
                        t.ind = subs;
                        t.var = b;
                    end
                    m = max(t.ind, [], 1);
                    t.sz(m > t.sz) = m(m > t.sz);
                otherwise
                    error('Only () assignment is supported for Tensorinos')
            end
        end

        function t_sub = subsref(t, s)
            % Subscripted reference for a :class:`Tensorino`.
            %
            % Todo
            % ----
            % Fix case where all dimensions are indexed, fix indexing with ranges, write
            % docstring.
            %
            % Usage
            % -----
            % :code:`a_sub = a(i1, i2, ..., iN)`
            %   where each :code:`in` is an integer index, returns a scalar.
            %
            % :code:`a_sub = a(R1, R2, ..., RN)`
            %   where each :code:`Rn` is either a colon, an array of integers representing a
            %   slice in this dimension or a logical array representing the logical indexing
            %   of this dimension. Returns a sparse array.
            %
            % Example
            % -------
            % .. code-block:: matlab
            %
            %   >> t = Tensorino([4, 4, 4; 2, 2, 1; 2, 3, 2], [3; 5; 1], [4, 4, 4]);
            %   >> t(1, 2, 1) %<-- returns zero
            %   >> t(4, 4, 4) %<-- returns 3
            %   >> t(2, :, :) %<-- returns a 1 x 4 x 4 Tensorino
            switch s(1).type
                case '()'
                    n = size(s.subs, 2);
                    assert(n == ndims(t), 'Number of indexing indices must match tensor size')
                    f = true(size(t.ind, 1), 1);
                    sz = zeros(1, size(s.subs, 2));
                    for i = 1:size(s.subs, 2)
                        if strcmp(s.subs{i}, ':')
                            sz(i) = t.size(i);
                            continue;
                        end
                        A = s.subs{i};
                        if length(A) ~= length(unique(A))
                            error('Repeated index in position %i', i);
                        end
                        if ~isempty(t.ind)
                            B = t.ind(:, i);
                            P = false(max(max(A), max(B))+1, 1);
                            P(A + 1) = true;
                            f = and(f, P(B+1));
                        [~, ~, temp] = unique([A(:); t.ind(f, i)], 'stable');
                        t.ind(f, i) = temp(length(A) + 1:end);
                        sz(i) = length(A);
                        end
                    end
                    if isempty(t.ind)
                        t_sub = Tensorino([], [], sz);
                    else
                        t_sub = Tensorino(t.ind(f, :), t.var(f), sz);
                    end
                    if all(cellfun(@isnumeric, s.subs) & cellfun(@(x) length(x) == 1, s.subs))
                        t_sub = full(t.var(f));
                    end
                otherwise
                    % only overload parentheses indexing
                    t_sub = builtin('subsref', t, s);
            end
        end
        
        function s = sum(t)
            % Sum of all elements of a :class:`Tensorino`.
            s = full(sum(t.var));
        end
        
        function t = symp(t)
            % Symbolic something
            %
            % Todo
            % ----
            % Figure out what this does.
            t.var = symp(t.var);
        end
        
        function C = tensorprod(A, B, dimA, dimB, options)
            % Tensor products between two :class:`Tensorino`s.
            %
            % Usage
            % -----
            % :code:`C = tensorprod(A, B, dimA, dimB)`
            %	returns the tensor product of tensors :code:`A` and :code:`B`. The arguments
            %   :code:`dimA` and :code:`dimB` are vectors that specify which dimensions to
            %   contract in :code:`A` and :code:`B`. The size of the output tensor is the
            %   size of the uncontracted dimensions of :code:`A` followed by the size of the
            %   uncontracted dimensions of :code:`B`.
            %
            % :code:`C = tensorprod(A, B)`
            %	returns the outer product of tensors :code:`A` and :code:`B`. This is
            %   equivalent to the previous syntax with
            %   :code:`dimA` = :code:`dimB` = :code:`[]`.
            %
            % :code:`C = tensorprod(_, 'NumDimensionsA', ndimsA)`
            %	optionally specifies the number of dimensions in tensor :code:`A` in
            %   addition to combat the removal of trailing singleton dimensions.

            arguments
                A Tensorino
                B Tensorino
                dimA = []
                dimB = []
                options.NumDimensionsA = ndims(A)
            end

            szA = size(A, 1:options.NumDimensionsA);
            szB = size(B, 1:max(ndims(B), max(dimB)));

            assert(length(dimA) == length(dimB) && all(szA(dimA) == szB(dimB)), ...
                'Tensorino:dimerror', 'incompatible contracted dimensions.');

            uncA = 1:length(szA); uncA(dimA) = [];
            uncB = 1:length(szB); uncB(dimB) = [];
            
            szC = [szA(uncA) szB(uncB)];

            if isempty(A.ind) || isempty(B.ind)
                C = Tensorino([], [], szC);
                return;
            end

            [uniqA, ~, icA] = unique(A.ind(:, uncA), 'rows');
            [uniqB, ~, icB] = unique(B.ind(:, uncB), 'rows');
            [~, ~, icC] = unique([A.ind(:, dimA); B.ind(:, dimB)], 'rows');
            
            % TODO: unify symbolic and double case, if possible
            if isa(A.var, 'double')
                if numel(A.var) / max(icA) / max(icC) < 0.1
                    m1 = sparse(icA, icC(1:numel(A.var)), A.var, max(icA), max(icC));
                else
                    m1 = full(sparse(icA, icC(1:numel(A.var)), A.var, max(icA), max(icC)));
                end
            else
                m1 = sym(zeros(max(icA), max(icC)));
                m1(sub2ind_([max(icA), max(icC)], icA, icC(1:numel(A.var)))) = A.var;
            end
            if isa(B.var, 'double')
                if numel(B.var) / max(icB) / max(icC) < 0.1
                    m2 = sparse(icC(numel(A.var)+1:end), icB, B.var, max(icC), max(icB));
                else
                    m2 = full(sparse(icC(numel(A.var)+1:end), icB, B.var ,max(icC), max(icB)));
                end
            else
                m2 = sym(zeros(max(icC), max(icB)));
                m2(sub2ind([max(icC), max(icB)], icC(numel(A.var)+1:end), icB)) = B.var;
            end

            m = m1 * m2;
            [r, c, v] = find(m);
            
            % scalar result
            if isempty(uncA) && isempty(uncB)
                C = v;
                return
            end
            
            % tensor results
            ind = cat(2, uniqA(r, :), uniqB(c, :));
            C = Tensorino(ind, v(:), szC);
        end
        
        function t = times(a, b)
            % Product of a :class:`Tensorino` and a scalar.
            %
            % Usage
            % -----
            % :code:`t = times(t, a)`
            %
            % :code:`t = a .* t`                    
            assert(isscalar(a) || isscalar(b), 'Only multiplication of a Tensorino by a scalar is supported');
            if isscalar(a)
                t = Tensorino(b.ind, a*b.var, b.size);
            else
                t = Tensorino(a.ind, b*a.var, a.size);
            end
        end
        
        function t = transpose(t)
            % Transpose.
            %
            % Only defined for 2D :class:`Tensorino`.
            %
            % Usage
            % -----
            % :code:`a = transpose(t)`
            %
            % :code:`a = t.'`
            %
            % Arguments
            % ---------
            % t : :class:`Tensorino`
            %   input tensor.
            %
            % Returns
            % -------
            % a : :class:`Tensorino`
            %   output tensor.
            assert(ismatrix(t), 'Tensorino:RankError', 'ctranspose is only defined for rank 2 Tensorinos.');
            t = permute(t, [2, 1]);
        end

        function t = uminus(t)
            % Unary minus.
            %
            % Usage
            % -----
            % :code:`b = uminus(a)`
            %
            % :code:`b = -a`
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array.
            t.var = -t.var;
        end
        
        function type = underlyingType(t)
            type = underlyingType(t.var);
        end

        function a = uplus(a)
            % Unary plus.
            %
            % Usage
            % -----
            % :code:`b = uplus(a)`
            %
            % :code:`b = +a`
            %
            % Arguments
            % ---------
            % a : :class:`Tensorino`
            %   input array.
            %
            % Returns
            % -------
            % b : :class:`Tensorino`
            %   output array.
            return;
        end
        
    end
    
    methods (Static)
        
        function t = delta(numinds, inddim)
            % Create delta- (ghz-) tensor with given number of indices and index dimension.
            %
            % Arguments
            % ---------
            % numinds : :class:`int`
            %   Number of indices of delta-array.
            %
            % inddim : :class:`int`
            %   Dimension of each index of delta-array.
            %
            % Returns
            % -------
            % t : :class:`Tensorino`
            %   output delta-array.
            t = Tensorino(repmat(1:inddim, numinds, 1)', 1, repmat(inddim, 1, numinds));
        end
        
        function t = new(fun, sz, density)
            % Create a :class:`Tensorino` with data generated using a function handle.
            %
            % Arguments
            % ---------
            % fun : :class:`function_handle`
            %   Function of signature :code:`fun(dims)`.
            %   If this is left empty, the tensor data will be uninitialized.
            %
            % sz : (1, :) :class:`int`
            %   Size of the sparse tensor.
            %
            % density : :class:`double`
            %   Density of nonzero elements (0 < :code:`density` < 1).
            %
            % Returns
            % -------
            % t : :class:`Tensorino`
            %   Output tensor.
            if any(sz > 2^32-1)
                error("Individual index size must be smaller than 2^32-1");
            end
            nz = floor(prod(sz * density^(1/numel(sz))));
            ind = ceil(rand(nz, length(sz)) * diag(sz));
            ind = unique(ind, 'rows');
            var = fun(size(ind, 1), 1);
            t = Tensorino(ind, var, sz);
        end
        
        function t = rand(sz, density)
            t = Tensorino.new(@rand, sz, density);
        end
        
        function t = randn(sz, density)
            t = Tensorino.new(@randn, sz, density);
        end
        
        function t = randc(sz, density)
            t = Tensorino.new(@randc, sz, density);
        end
        
        function t = randnc(sz, density)
            t = Tensorino.new(@randnc, sz, density);
        end
        
    end
    
end

%% Internal auxiliary functions
function s = dim2str(sz)

s = regexprep(mat2str(sz), {'\[', '\]', '\s+'}, {'', '', 'x'});

end

function I = ind2sub_(sz, ind, perm)
% Faster implementation of builtin ind2sub

if isempty(ind)
    I = [];
    return;
elseif size(sz,2) == 1
    I = ind;
    return;
end

if nargin<3
    perm=1:numel(sz);
else
    perm(perm)=1:numel(sz);
end

nout = numel(sz);
I = zeros(numel(ind),numel(sz));

if nout > 2
    k = cumprod(sz);
    for i = nout:-1:3
        I(:,perm(i)) = floor((ind-1)/k(i-1)) + 1;
        ind = rem(ind-1, k(i-1)) + 1;
    end
end

if nout >= 2
    I(:,perm(2)) = floor((ind-1)/sz(1)) + 1;
    I(:,perm(1)) = rem(ind-1, sz(1)) + 1;
else 
    I(:,perm(1)) = ind;
end

end

function ind = sub2ind_(sz,I)
% Faster implementation of builtin sub2ind.

if isempty(I)
    ind = [];
    return;
elseif size(I, 2) == 1
    ind = I;
    return;
end

numOfIndInput = size(I, 2);

ind = I(:, 1);
if numOfIndInput >= 2
    % compute linear indices
    ind = ind + (I(:, 2) - 1).*sz(1);
end 
    
if numOfIndInput > 2
    % compute linear indices
    k = cumprod(sz);
    for i = 3:numOfIndInput
        ind = ind + (I(:, i) - 1) * k(i - 1);
    end
end

end

function sub2 = sub2sub_(sz2, sz1, sub1)
% Convert subscript to subscript without converting to a linear index

if isempty(sub1) || isequal(sz1, sz2)
    sub2 = sub1;
    return;
end
sub2 = zeros(size(sub1, 1), numel(sz2)); % preallocate new subs
nto1 = sum(cumsum(flip(sz1-1), 2) == 0, 2); % extract number of trailing ones in both sizes
nto2 = sum(cumsum(flip(sz2-1), 2) == 0, 2);
pos1_prev = 0;  pos2_prev = 0;
flag = true;
while flag
    [pos1, pos2] = find(cumprod(sz1(pos1_prev+1:end)).' == cumprod(sz2(pos2_prev+1:end)), 1);
    pos1 = pos1 + pos1_prev;
    pos2 = pos2 + pos2_prev;
    if prod(sz1(pos1_prev+1:pos1)) > 2^32-1
        error('Cannot map subscripts to new size as intermediate index exceeds MAXSIZE.')
    end
    sub2(:, pos2_prev+1:pos2) = ind2sub_(sz2(pos2_prev+1:pos2), sub2ind_(sz1(pos1_prev+1:pos1), sub1(:, pos1_prev+1:pos1)));
    if pos2 == numel(sz2) - nto2 || pos1 == numel(sz1) - nto1
        flag = false;
    else
        pos1_prev = pos1;
        pos2_prev = pos2;
    end
end
sub2(:, end-nto2+1:end) = 1;
end
