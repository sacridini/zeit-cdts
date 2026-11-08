## A literal replica of the scsi computation inside DetectAbsSnow.m of Fmask 4.7 (https://github.com/GERSL/Fmask,
## commit e32f5bd, MIT License, Copyright (c) 2018 Global Environmental Remote Sensing Lab; lines 41-88), returning the
## scsi that the .m keeps inside (its output is only scsi < 9 & psnow & ~saturated), so that the values can be compared
## with Zeit's and the class of the product seen.
function [scsi, cls] = fm_scsi (band_green, ndsi, psnow, resolution)
    band_green_filled = fillmissing(band_green,'constant',0);
    radius = fix(5000/resolution);
    win_size=2*radius+1;
    num_clips = 5;
    [size_row, size_colum] =size(band_green_filled);
    num_per_row = round((size_row)/num_clips);

    row_nums = zeros([1,num_clips],'double');
    row_nums(1:num_clips-1)=num_per_row;
    row_nums(num_clips) = size_row - num_per_row*(num_clips-1);

    scsi = zeros([size_row, size_colum],'double');
    for i_row = 1:num_clips
        row_ed = sum(row_nums(1:i_row));
        row_st = row_ed - row_nums(i_row) + 1;

        row_st_exp = row_st - radius;
        row_ed_exp = row_ed + radius;
        row_st_small = radius + 1;
        row_ed_small = radius + row_nums(i_row);
        if row_st_exp < 1
            row_st_exp =1;
            row_st_small =1;
            row_ed_small = row_nums(i_row);
        end
        if row_ed_exp > size_row
            row_ed_exp = size_row;
        end
        scsi_small= stdfilt(band_green_filled(row_st_exp:row_ed_exp,:),...
                 true(win_size)).*(1-ndsi(row_st_exp:row_ed_exp,:));
        cls = class (stdfilt(band_green_filled(row_st_exp:row_st_exp,:), true(3)).*(1-ndsi(row_st_exp:row_st_exp,:)));
        scsi(row_st:row_ed,:) = scsi_small(row_st_small:row_ed_small,:);
        clear scsi_small;
    end
endfunction
