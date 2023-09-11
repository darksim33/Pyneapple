%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   Multi-Exponential Signal Map Generator
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% General information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tool to generate synthetic multi-exponential DWI signals for different
% structures organised in a multi-slice 4D NII format.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_pixel = 176;
SNR = [60, 100, 140]; % SNR per slice
n_slices = length(SNR);
b = [0,5,10,20,30,40,50,75,100,150,200,250,300,400,525,750];

% Specifying diffusion components for different structures
f_1 = [60 30 10];
d_1 = [1 5.8 165].*1e-3; 
f_2 = [25 75 5]; 
d_2 = [2 7 200].*1e-3; 

fdInput = [d_1 f_1;...
           d_2 f_2];

input = table(n_pixel, n_slices, SNR, fdInput(1,:), fdInput(2,:), b);

% Allocate data structure for pixels signals
signalMap = zeros(n_pixel, n_pixel, n_slices, length(b));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create background structure determining components diffusion parameters
structure = [ones(n_pixel, n_pixel/2), ones(n_pixel, n_pixel/2)*2];

for slice = 1:n_slices
    for i = 1:n_pixel
        for j = 1:n_pixel

            % Create multi-exponential signal based on input parameters
            rawSignal = createSignal(fdInput(structure(i,j), :), b);
        
            % Generating noise for every pixel
            [signal, noise] = noiseSignal(rawSignal, SNR(slice));
    
            signalMap(i,j,slice,:) = signal;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create NII and saving data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert simulation data to NII format
niftiwrite(signalMap, "tri-exp_gT_map.nii");

% Write simulation data to file
writetable(input, "input.xlsx"); 