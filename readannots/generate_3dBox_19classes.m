addpath('./jsonlab')
addpath('./mBB')
addpath('~/workspace/bin/faster-rcnn/tools/') % to add writePC2XYZfile.m to path (PC, path, output name)
addpath('readData/')


relative_idx = fopen('relative_num.txt','w');
fprintf(relative_idx, 'original_number new_number\n');

cnt=1;
for imagenum = 1:1449
    disp(fprintf('image number: %d\n',imagenum))
    if ~exist('alignData','dir')
        mkdir('alignData')
    end
    scenefile = sprintf('image%04d',imagenum);
    pathtoscenefile = sprintf('manhattan_bfx_alignData_with_nan_19_classes/%s',scenefile);
    if ~exist(pathtoscenefile,'dir')
        mkdir(pathtoscenefile)
    end    
    thispath =sprintf('/home/closerbibi/workspace/data/NYUdata/NYU%04d',imagenum);
    [data,bb3d] = readframeSUNRGBD(thispath,'');

%     save(sprintf('%s/pc.mat',pathtoscenefile),'points3d','-v7.3')
    %%% closed
    %writePC2XYZfile(points3d(1:10:end,:), pathtoscenefile, 'pc')
%%%    annofile = sprintf('~/bin/faster-rcnn/data/DIRE/Annotations/picture_%06d.txt',count);
%%%    AnnotationID = fopen(annofile,'w');  %% not align with the point cloud

% % % %% get z rotation for manhattan assumption
% % %     dir = data.Rtilt*[0;1;0]; % vector rotate from y direction to Rtilt
% % %     angle = dir(1:2,1)/norm(dir(1:2,1)); % unit vector of x-y plane
% % % %     if (angle(2)<0.9)
% % %     R = getRotationMatrix('z',atan2(angle(1),angle(2))); % atan2: Convert Complex Number to Polar Coordinates(ex: 3+5i to theta)
% % %     R = R(1:3,1:3);

%% find R to align
%     R = findalignR(points3d);

%%
    original_corners = {};
    cls_idx = 1;
    for i=1:length(bb3d)
        clsn = bb3d(i).classname;
        cls_lst{cnt} = clsn;
        cnt = cnt + 1;
        if strcmp(clsn,'sofa_chair')
            disp(imagenum)
        end
        if strcmp(clsn, 'bed')|| strcmp(clsn, 'chair')|| strcmp(clsn, 'table')|| strcmp(clsn, 'sofa')|| strcmp(clsn, 'toilet')|| ...
            strcmp(clsn, 'bathtub')|| strcmp(clsn, 'bookshelf')|| strcmp(clsn, 'box')|| strcmp(clsn, 'counter')|| strcmp(clsn, 'desk')|| ...
            strcmp(clsn, 'door')|| strcmp(clsn, 'dresser')|| strcmp(clsn, 'garbage_bin')|| strcmp(clsn, 'lamp')|| strcmp(clsn, 'monitor')|| ...
            strcmp(clsn, 'night_stand')|| strcmp(clsn, 'pillow')|| strcmp(clsn, 'sink')|| strcmp(clsn, 'tv')|| strcmp(clsn, 'nightstand')
            % television --> tv, garbagebin --> garbage_bin
            original_corners{cls_idx} = get_corners_of_bb3d(bb3d(i));     
            % new orientation
%             corners = original_corners{cls_idx}*R;
            corners = original_corners{cls_idx};
            xmin(cls_idx) = min(corners(1:4,1));
            xmax(cls_idx) = max(corners(1:4,1));
            ymin(cls_idx) = min(corners(1:4,2));
            ymax(cls_idx) = max(corners(1:4,2));

            zmin(cls_idx) = min(corners(1:8,3));
            zmax(cls_idx) = max(corners(1:8,3));

            clss{cls_idx} = clsn;
%             disp(clsn)
            %% write box to xyz file and log the original file
            %% closed
            %writePC2XYZfile(corners,pathtoscenefile,...
            %    sprintf('box_%03d_%s',i,bb3d(i).classname))
            cls_idx = cls_idx +1;
        end
    end
    
    %% debug
    %showalign(points3d, original_corners, R, xmin, ymin, zmin, xmax, ymax, zmax)
    
    
    %% new orientation    
    %points3d = points3d * R;
    
    close all
    
    if cls_idx == 1
%             save(sprintf('%s/annotation_pc.mat',pathtoscenefile),'points3d')
            clear clss
            clear xmin
            clear xmax
            clear ymin
            clear ymax
            clear zmin
            clear zmax
            clear original_corners
            continue
    end
%     save(sprintf('%s/annotation_pc.mat',pathtoscenefile),'points3d','xmin','xmax','ymin','ymax', 'zmin', 'zmax', 'clss')
    clear clss
    clear xmin
    clear xmax
    clear ymin
    clear ymax
    clear zmin
    clear zmax
    clear original_corners
    %%%    fclose(AnnotationID);
%     fprintf(relative_idx, '%d %d\n',imagenum,count);
%     count = count+1;
end
fclose(relative_idx);
cls_names = unique(cls_lst);
disp(cls_names)