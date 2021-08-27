for idx = 1:9
    n = idx
    eval(['addpath("MRCLAM_Dataset' num2str(n) '")'  ])

    run loadMRCLAMdataSet.m
    run sampleMRCLAMdataSet.m
    n = idx
    eval(['save("Processed_data/Dataset' num2str(n) '")'   ]);
    eval(['rmpath("MRCLAM_Dataset' num2str(n) '")'])
end