#for i in Xception VGG16 VGG19 ResNet50 InceptionV3; do
#	for j in -1 0 25 50 75 100; do
#		python script_1.py --architecture $i --fineTunningRate $j > ../results/"$i""_""$j"_doenca.txt
#		mv ../models_checkpoints/* /media/hd/gabriel/experimento10oct/
#	done
#done

for i in InceptionV3; do
	for j in -1 0 25 50 75 100; do
		python script_2.py --architecture $i --fineTunningRate $j > ../results/"$i""_""$j"_daninha.txt
		mv ../models_checkpoints/* /media/hd/gabriel/experimento10oct/
	done
done
