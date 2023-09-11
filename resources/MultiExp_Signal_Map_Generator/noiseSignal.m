function [signal, noise] = noiseSignal(rawSignal, SNR)

    % Generating noise (as overlay) new for every pixel
    rawNoise = randn(size(rawSignal)); % noise based on gaussian distribution of randn()
    noise_factor = rawSignal(1)/(SNR*std(rawNoise));
    noise = noise_factor*rawNoise;
    signal = rawSignal + noise_factor*rawNoise;
    
end