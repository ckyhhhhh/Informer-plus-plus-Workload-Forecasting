if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Informer_pp
data_path_name=observed_values.csv
model_id_name=workload
data_name=workload
random_seed=2021
epoch=30


#######################

root_path_name=2011/machine
seq_len=48
pred_len=24
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001
#
#
#
root_path_name=2011/machine
seq_len=96
pred_len=48
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001
#
#
root_path_name=2011/machine
seq_len=168
pred_len=168
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001
#
#
root_path_name=2011/machine
seq_len=168
pred_len=336
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


# #######################


root_path_name=2019/container
seq_len=48
pred_len=24
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001




root_path_name=2019/container
seq_len=96
pred_len=48
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2019/container
seq_len=168
pred_len=168
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2019/container
seq_len=168
pred_len=336
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001



# #######################



root_path_name=2018/container
seq_len=48
pred_len=24
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001



root_path_name=2018/container
seq_len=96
pred_len=48
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2018/container
seq_len=168
pred_len=168
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2018/container
seq_len=168
pred_len=336
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001



# #######################



root_path_name=2017/machine
seq_len=48
pred_len=24
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001



root_path_name=2017/machine
seq_len=96
pred_len=48
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2017/machine
seq_len=168
pred_len=168
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001


root_path_name=2017/machine
seq_len=168
pred_len=336
python -u run_longExp.py --random_seed $random_seed --is_training 1 --root_path $root_path_name --data_path $data_path_name --model_id $model_id_name_$seq_len'_'$pred_len --model $model_name --data $data_name --features M --seq_len $seq_len --pred_len $pred_len --enc_in 4 --e_layers 3 --n_heads 4  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' --train_epochs $epoch --itr 1 --batch_size 32 --learning_rate 0.0001



