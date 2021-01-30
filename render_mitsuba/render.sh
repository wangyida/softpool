root="../pytorch/pcds"
for categories in $(ls $root/gt)
do
	for folder in $(ls $root)
	do 
		echo $folder
		ls $root/$folder/$categories > $root/$folder/$categories/results.list
		python3 render_mitsuba2_pc.py $root/$folder/$categories/results.list
		rm -rf $root/$folder/$categories/results.list
		rm -rf $root/$folder/$catebories/*.exr
	done
done


