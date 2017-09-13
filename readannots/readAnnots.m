annotspath = '../dataset/NYUV2/annotations';

lst = dir(annotspath);

target_cls = {};
cnt = 1;
imgcnt = 0;
for i = 1:length(lst)-2
    disp(i)
    ano = load(sprintf('%s/%d.mat',annotspath,i));
    cls = ano.data.gtLabelNames;
    box = ano.data.gt3D;
    wantedindex = find(~cellfun(@isempty,box));
    if ~isempty(wantedindex)
        imgcnt = imgcnt+1;
    end
%     wantedindex = find(~cellfun(@isempty,box) | cellfun(@isempty,box)) ;
%     for j = 1:length(wantedindex)   
%         target_cls{cnt} = cls{wantedindex(j)};
%         cnt = cnt +1;
%     end
end   

% final_target_cls = unique(target_cls);
% final_target_cls = final_target_cls';