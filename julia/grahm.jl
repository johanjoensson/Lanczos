using LinearAlgebra

function classical_gram_schmidt_alt(matrix)
    # orthogonalises the columns of the input matrix
    num_vectors = size(matrix)[2]
    orth_matrix = zeros(size(matrix))
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = matrix[:, vec_idx]
        sum = zeros(size(orth_matrix[:, 1]))
        for span_base_idx = 1:(vec_idx-1)
            # compute sum
            sum += dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, span_base_idx]
        end
        orth_matrix[:, vec_idx] -= sum
        # normalise vector
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
    end
    return orth_matrix
end

function modified_gram_schmidt(matrix)
    # orthogonalises the columns of the input matrix
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx]/norm(orth_matrix[:, vec_idx])
        for span_base_idx = (vec_idx+1):num_vectors
            # perform block step
            orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx])*orth_matrix[:, vec_idx]
        end
    end
    return orth_matrix
end

function my_gram_schmidt(matrix)
    ort = similar(matrix)
    for i=1:size(matrix, 2)
        ort[:, i] .= matrix[:, i] - matrix[:, 1:i-1]*(matrix[:, 1:i-1]'*matrix[:, i])
        ort[:, i] = normalize(ort[:, i])
    end
    return ort
end
function my_modified_gram_schmidt(matrix)
    ort = similar(matrix)
    for i=1:size(matrix, 2)
        ort[:, i] = normalize(ort[:, i])
        ort[:, i+1:end] .= matrix[:, i+1:end] - matrix[:, i]*(matrix[:, i]'*matrix[:, i+1:end])
    end
    return ort
end

function P!(Q, q)
    q .= (I - Q*Q')q
end

function P_perp!(Q, q)
    for i=size(Q, 2):-1:1
        q .= (I - Q[:, i]*Q[:, i]')*q
    end
    q
end
