environment:
    CartPole-v2

agent:
    network:
        MLP[2, 256]
    
    optimizer:
        Adam[lr=3e-4]

    replay_buffer: 
        capacity: 10000
        batch_size: 32

    update_function:
        formula: "Q(s, a) += alpha * (reward + gamma * max(Q(next_state)) - Q(s, a))"
        parameters:
            alpha: learning_rate
            gamma: discount_factor


