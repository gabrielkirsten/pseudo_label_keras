for i in Xception VGG16 VGG19 ResNet50 InceptionV3; do
	for j in -1 20 20 30 40 50 60 70 80; do
		nohup python main.py -d /home/aluno/gabriel/_dataset/doencas/doencas3000/ -n $j -a $i 
		# mv ../models_checkpoints/* /media/hd/gabriel/experimento10oct/
	done
done
