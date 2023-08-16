function [rawSignal] = createSignal(fdInput, b)

    n_compartments = length(fdInput)/2;
    d = fdInput(1:n_compartments);
    f = fdInput(n_compartments+1:end);

    % Define total diffusion signal decay according to sum of f_i*e^(-b*D_i)
    A = exp(-kron(b', d)); % constraint matrix containing exp decay funxtions
    rawSignal = A*f';      % signal of voxel without noise


end
