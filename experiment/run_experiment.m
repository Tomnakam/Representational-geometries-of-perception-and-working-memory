function PRMdecoding

try
    %% Preparation
    AssertOpenGL;
    KbName('UnifyKeyNames');
    [keyboardIdx, productNames] = GetKeyboardIndices;
    load('PRM_OverallParameters.mat');

    id=input('id=');
    run=input('run=');
    test=input('If you need to do a button test, Press "1".');

    %% Direcotry
    olddir1=pwd;
    cd('rawdata');

    folder_name=['sub',num2str(id)];
    cd(folder_name);
    load(['rawdata_PRMdecoding_exp1_',num2str(id)],'trial');
    cd(olddir1);

    %ListenChar(2);
    Screen('Preference','SyncTestSettings',0.001);

    white=[255 255 255];
    black=[0 0 0];
    gray=(white+black)/2;

    screens=Screen('Screens');
    if ispc %windows
        disp('This is a Windows OS.');
        screenNumber=1;
    elseif ismac %mac
        disp('This is a Mac OS.');
        screenNumber=max(screens);
    elseif isunix %linux
        disp('This is a Linux OS.');
        disp('Please be careful about monitor settings.');
        screenNumber=max(screens);
    else
        error('Unknown OS.');
    end

    if isscalar(screens) %single monitor setting ****only for test
        [w,rect]=Screen('Openwindow', screenNumber, gray, [0 0 800 600]);
    else
        [w,rect]=Screen('Openwindow', screenNumber, gray);
    end
    Priority(MaxPriority(w));
    Screen('Flip',w);
    
    [cx, cy] = RectCenter(rect);
    width = Screen('WindowSize', w);
    ppd=width*pi/atan(width_cm/view_dist/2)/360; % pixel per degree

    [xcenter,ycenter]=RectCenter(rect);
    Screen('BlendFunction',w,'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

    %% Temporal parameters
    ifi = Screen('GetFlipInterval', w); % the length of each frame ~ 16ms
    numFrames = round(1 / ifi); %Hz

    %% Spatial paramters
    mask_rect=[-128 -128 128 128]*2.5;
    mask_loc=[xcenter, ycenter, xcenter, ycenter]+mask_rect;
    stim_rect=[-128 -128 128 128]*2.5;
    stim_loc=[xcenter, ycenter, xcenter, ycenter]+stim_rect;

    Screen('TextSize',w,textsize);
    fprintf('\n sample/test width: %.2f degree of visual angle',(stim_rect(3)-stim_rect(1))/ppd);
    fprintf('\n mask width: %.2f degree of visual angle',(mask_rect(3)-mask_rect(1))/ppd);
    fprintf('\n text height: %.2f degree of visual angle',textsize/ppd);

    events = table(repmat("None",768,1),zeros(768,1),zeros(768,1),...
        'VariableNames',{'event','onset','duration'});

    event_nb = 0;

    %% Read images

    %read face images
    cd(olddir1);
    cd('rawdata');
    cd(folder_name);
    cd('stimuli');

    for n = 1:16
        filename = sprintf('sample_run%d_trial%d.png', run, n);
        blended_tex_sample(n) = Screen('MakeTexture', w, imread(filename));
        filename = sprintf('test_run%d_trial%d.png', run, n);
        if isfile(filename) 
            blended_tex_test(n)=Screen('MakeTexture', w, imread(filename));
        end
    end
    
    %read mask image
    cd(olddir1);
    load('colored_noise.mat','colored_noise');
    for j=1:1
        mask_tex(j)=Screen('MakeTexture',w,127.5*colored_noise(:,:,:,j)+127.5);
    end

    cd('rawdata');

    %% Test Buttons
    if test==1
        UseKeys={sameKey,diffKey};
        DrawFormattedText(w, 'Testing buttons...',  'center', 'center', black);
        Screen('Flip',w);
        WaitSecs(3);
        for j=1:length(UseKeys)
            DrawFormattedText(w, ['Press Key ' num2str(cell2mat(UseKeys{j}))], 'center', 'center', black);
            Screen('Flip',w);
            fprintf('\n Waiting for the partcipant to press Key %d.\n', j);

            while true
                [keyIsDown,reaction,KeyCode] = KbCheck;
                if keyIsDown
                    if ismember(UseKeys{j},string(KbName(KeyCode)))
                        Screen('Flip',w);
                        break
                    end
                end
            end
            WaitSecs(1);
        end
        Screen('Flip',w);

    end

    %% Start trials
    HideCursor(w);

    fprintf('\n Everything is set! Press enter to start the experiment.\n');
    DrawFormattedText(w, 'Get Ready', 'center', 'center', black);
    Screen('Flip',w);

    while true
        [keyIsDown,secs,keyCode] = KbCheck;
        if keyIsDown
            if strcmp(KbName(keyCode),'Return')
                break
            end
        end
    end

    fprintf('\n Ask the technician to start the scan.\n');
    volume = 0;

    while volume<5 % wait for the 5 warmup triggers
        [keyIsDown,rt,keyCode]=KbCheck;
        if keyIsDown
            if volume==1
                fprintf('\n Waiting for the scanner to reach equilibrium.\n');
                Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
                Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
                Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
                Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
                Screen('Flip',w);
            end
            if strcmp(KbName(keyCode),'s')
                run_start=GetSecs;
                volume=volume+1;
                disp(volume);
                WaitSecs(0.05);
            end
        end
    end

    cd(folder_name);
    for n=1:16

        keyIsDown=0;

        %cue
        for j=1:round(cue_duration/ifi)
            if trial(n,2,run)==1 %face cued
                DrawFormattedText(w, 'FACE', 'center', 'center', black);
            elseif trial(n,2,run)==2 %scene cued
                DrawFormattedText(w, 'SCENE', 'center', 'center', black);
            end
            if j==1
                [flp,cue_onset_1]=Screen('Flip',w);
                event_nb = event_nb+1;
                events.event(event_nb)="Cue";
                events.onset(event_nb)=cue_onset_1-run_start;
            else
                [flp,cue_onset]=Screen('Flip',w);
            end
        end
        
        %encoding
        for j=1:delay-cue_duration
            for i=1:round(sample_duration/ifi)
                Screen('DrawTexture',w,blended_tex_sample(n),[],stim_loc);
                Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
                Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
                Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
                Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
                [flp,sample_onset]=Screen('Flip',w);
                if i==1
                    event_nb = event_nb+1;
                    if j==1
                        events.event(event_nb)="FirstSample";
                    else
                        events.event(event_nb)="Sample";
                    end
                    events.onset(event_nb)=sample_onset-run_start;
                end
            end

            if j<delay-cue_duration
                for i=1:round(blank_duration/ifi)
                    Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
                    Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
                    Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
                    Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
                    [flp,sample_offset]=Screen('Flip',w);
                    if i==1
                        event_nb = event_nb+1;
                        if j==delay-cue_duration
                            events.event(event_nb)="LastBlank";
                        else
                            events.event(event_nb)="Blank";
                        end
                        events.onset(event_nb)=sample_offset-run_start;
                    end
                end

            else
                %mask
                for i=1:round(blank_duration/ifi)
                    Screen('DrawTexture',w,mask_tex,[],mask_loc);
                    Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
                    Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
                    Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
                    Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
                    [flp,sample_offset]=Screen('Flip',w);
                    if i==1
                        event_nb = event_nb+1;
                        if j==delay-cue_duration
                            events.event(event_nb)="LastBlank";
                        else
                            events.event(event_nb)="Blank";
                        end
                        events.onset(event_nb)=sample_offset-run_start;
                    end
                end
            end
        end
        
        %delay
        for i=1:round(delay/ifi)
            Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
            Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
            Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
            Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
            [flp,mask_offset]=Screen('Flip',w);
            if i==1
                event_nb = event_nb+1;
                events.event(event_nb)="Delay";
                events.onset(event_nb)=mask_offset-run_start;
            end
        end

        %test & response
        if trial(n,8,run)==0
            curr_resp="None";
        elseif trial(n,8,run)==1
            Screen('DrawTexture',w,blended_tex_test(n),[],stim_loc);
            Screen('FillOval', w, black, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
            Screen('DrawLine', w, white, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
            Screen('DrawLine', w, white, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
            Screen('FillOval', w, black, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
            [flp,test_onset]=Screen('Flip',w);
            event_nb = event_nb+1;
            events.event(event_nb)="Test";
            events.onset(event_nb)=test_onset-run_start;

            curr_resp="No";
            while GetSecs-test_onset<resp_time_limit-ifi
                [keyIsDown,rt,keyCode]=KbCheck(keyboardIdx);
                if keyIsDown
                    if strcmp(KbName(keyCode),sameKey) %same
    
                        if (trial(n,2,run)==1 && trial(n,9,run)==trial(n,4,run)) || (trial(n,2,run)==2 && trial(n,10,run)==trial(n,6,run))
                            trial(n,11,run)=1;
                            curr_resp="CR";
                        else
                            trial(n,11,run)=2;
                            curr_resp="Miss";
                        end
                        trial(n,12,run)= rt - test_onset;
    
                    elseif strcmp(KbName(keyCode),diffKey) %different
    
                        if (trial(n,2,run)==1 && trial(n,9,run)~=trial(n,4,run)) || (trial(n,2,run)==2 && trial(n,10,run)~=trial(n,6,run))
                            trial(n,11,run)=3;
                            curr_resp="Hit";
                        else
                            trial(n,11,run)=4;
                            curr_resp="FA";
                        end
                        trial(n,12,run)= rt - test_onset;
    
                    elseif strcmp(KbName(keyCode),'ESCAPE')
                        save(['rawdata_PRMdecoding_exp1_',num2str(id)],'trial');
                        error('"ESCAPE" was pushed');
                    end
                end
            end
        end

        %ITI
        Screen('FillOval', w, white, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
        Screen('DrawLine', w, black, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
        Screen('DrawLine', w, black, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
        Screen('FillOval', w, white, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
        [flp,ITI_onset]=Screen('Flip',w);
        event_nb = event_nb+1;
        events.event(event_nb)="ITI";
        events.onset(event_nb)=ITI_onset-run_start;
        while GetSecs-ITI_onset<trial(n,7,run)-ifi
            if keyIsDown==0
                [keyIsDown,rt,keyCode]=KbCheck(keyboardIdx);
                if keyIsDown
                    if strcmp(KbName(keyCode),sameKey) %same
    
                        if (trial(n,2,run)==1 && trial(n,9,run)==trial(n,4,run)) || (trial(n,2,run)==2 && trial(n,10,run)==trial(n,6,run))
                            trial(n,11,run)=1;
                            curr_resp="CR";
                        else
                            trial(n,11,run)=2;
                            curr_resp="Miss";
                        end
                        trial(n,12,run)= rt - test_onset;
    
                    elseif strcmp(KbName(keyCode),diffKey) %different
    
                        if (trial(n,2,run)==1 && trial(n,9,run)~=trial(n,4,run)) || (trial(n,2,run)==2 && trial(n,10,run)~=trial(n,6,run))
                            trial(n,11,run)=3;
                            curr_resp="Hit";
                        else
                            trial(n,11,run)=4;
                            curr_resp="FA";
                        end
                        trial(n,12,run)= rt - test_onset;
    
                    elseif strcmp(KbName(keyCode),'ESCAPE')
                        save(['rawdata_PRMdecoding_exp1_',num2str(id)],'trial');
                        error('"ESCAPE" was pushed');
                    end
                end
            end    
        end
        
        %displaying the result for each trial
        fprintf('\nRun %d Trial %d: %s.\n',run,n,curr_resp);

    end
    Screen('FillOval', w, white, [cx-d_out/2*ppd, cy-d_out/2*ppd, cx+d_out/2*ppd, cy+d_out/2*ppd], d_out*ppd);
    Screen('DrawLine', w, black, cx-d_out/2*ppd, cy, cx+d_out/2*ppd, cy, d_in*ppd);
    Screen('DrawLine', w, black, cx, cy-d_out/2*ppd, cx, cy+d_out/2*ppd, d_in*ppd);
    Screen('FillOval', w, white, [cx-d_in/2*ppd, cy-d_in/2*ppd, cx+d_in/2*ppd, cy+d_in/2*ppd], d_in*ppd);
    [flp,run_end]=Screen('Flip',w);
    event_nb = event_nb+1;
    events.event(event_nb)="FinalFlip";
    events.onset(event_nb)=run_end-run_start;
    for event_nb=1:height(events)-1
        events.duration(event_nb)=events.onset(event_nb+1)-events.onset(event_nb);
    end

    save(['rawdata_PRMdecoding_exp1_',num2str(id)],'trial');
    
    %displaying the number of correct response
    CRtrials=sum(mod(trial(1:16,11,run)',2),'omitnan');
    fprintf('\n Correct Response %d out of 8 trials',CRtrials);
    fprintf('\n Run is finished. Now waiting for 10 more triggers.\n');
    volume=0;
    while volume<10 % wait for the 9 more triggers
        [keyIsDown,rt,keyCode]=KbCheck;
        if keyIsDown
            if strcmp(KbName(keyCode),'s')
                volume=volume+1;
                disp(volume);
                WaitSecs(0.05);
            end
        end
    end

    DrawFormattedText(w, 'Good <- 1 2 3 4 -> Tired', 'center', 'center', black);
    Screen('Flip',w);
    while 1
        [keyIsDown,rt,keyCode]=KbCheck(keyboardIdx);
        if keyIsDown
            if ismember(KbName(keyCode),{'1!','2@','3#','4$'})
                fprintf('Tiredness:%s',KbName(keyCode));
                break
            end
        end
    end

    %creating timing files for Nilearn
    rowsCue=contains(events.event,"Cue");
    cue_timing=table2array(events(rowsCue,2:3));
    rowsTest=contains(events.event,"Test");
    testresp_timing=table2array(events(rowsTest,2:3));

    rowsSample=contains(events.event,"FirstSample");
    rowsDelay=contains(events.event,"Delay");
    P_timing=[table2array(events(rowsSample,2)),table2array(events(rowsDelay,2))-table2array(events(rowsSample,2))];
    WM_timing=table2array(events(rowsDelay,2:3));

    PFace_timing=P_timing(trial(:,2,run)==1,:);
    PScene_timing=P_timing(trial(:,2,run)==2,:);
    WMFace_timing=WM_timing(trial(:,2,run)==1,:);
    WMScene_timing=WM_timing(trial(:,2,run)==2,:);
    
    cueFtiming=[];cueStiming=[];
    for i=1:16
        if trial(i,2,run)==1
            cueFtiming=[cueFtiming;cue_timing(i,:)];
        else
            cueStiming=[cueStiming;cue_timing(i,:)];
        end
    end
   
    timing = table([cueFtiming(:,1);cueStiming(:,1);WMFace_timing(:,1);WMScene_timing(:,1);testresp_timing(:,1)], ...
        [PFace_timing(:,2)+cueFtiming(:,2);PScene_timing(:,2)+cueStiming(:,2);WMFace_timing(:,2);WMScene_timing(:,2);testresp_timing(:,2)], ...
        [repmat(["PFace"],8,1);repmat(["PScene"],8,1);repmat(["WMFace"],8,1);repmat(["WMScene"],8,1);repmat(["TestResp"],8,1)],...
        'VariableNames',{'onset','duration','trial_type'});
    writetable(timing,['event_sub',num2str(id),'_run',num2str(run),'.csv']);

    cd(olddir1);
    ShowCursor(w);
    Priority(0);
    Screen('Close');
    Screen('CloseAll');
    ListenChar(0);

catch
    cd(olddir1);
    ShowCursor(w);
    Priority(0);
    Screen('Close');
    Screen('CloseAll');
    psychrethrow(psychlasterror);
    ListenChar(0);
end

end