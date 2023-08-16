function write3DMatrixToTxt(matrix, filename)

    A = size(matrix);
    A(end+1)=1; % for missing dimension if numberOfIter == 1
    fileID = fopen(filename, 'w');
    
    for i = 1:A(3)
        fprintf(fileID,sprintf('layer %g:\n', i)); % mark different layers
        for j = 1:A(1)
            fprintf(fileID,'%g\t',matrix(j,:,i)); % print 2D matrix at layer i
            fprintf(fileID,'\n');
        end
        fprintf(fileID,'____________________________________\n \n'); 
    end
    fclose(fileID);
end