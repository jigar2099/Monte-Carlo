module Tools

# export doubler

function get_set_of_arrays(sample_size,examples,pulse_no,file_name)
    """
    This function takes required sample size and desired
    number of examples and returns empty arrays for MC data
    creation
    """
    full_shapes = np.load(file_name);
    full_shapes = reshape(full_shapes,sample_size,floor(Int,length(full_shapes)/sample_size));
    full_shapes = transpose(full_shapes);
    
    data = zeros(Int8,examples,sample_size)
    target = zeros(Int8,examples,sample_size)
    place= rand(1:sample_size,examples,pulse_no)
    selected_shapes = rand(1:size(full_shapes,1),examples,pulse_no)
    sel_sh = []
    for i in 1:size(selected_shapes,1)
        sh=selected_shapes[i,:]
        s = [filter(>(0),full_shapes[i,:]) for i in sh]
        push!(sel_sh,s)
        println(s)
    end
    return data, target, place, sel_sh
end
#sample_size=100;
#examples=2;
#pulse_no=5
#file_name = "uniq_samp_all.npy"
#a1,a2,a3,a4=get_set_of_arrays(sample_size,examples,pulse_no,file_name);# 



















function doubler(n)
    return 2 * n
end

function trippler(n)
    return 3 * n
end

end