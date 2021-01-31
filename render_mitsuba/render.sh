
root="../pytorch/pcds"
for outputx in $(ls $root)
do
	for categories in $(ls $root/gt)
	do
		echo $root/$outputx/$categories
		ls $root/$outputx/$categories/*.pcd > results.list
		python3 render_mitsuba2_pc.py results.list 
		rm -rf results.list
		rm -rf $root/$outputx/$catebories/*.exr
		rm -rf $root/$outputx/$catebories/*.xml
	done
done

: '
root="../pytorch/benchmark/pcds"
for outputx in $(ls $root)
do
	for categories in $(ls $root/gt)
	do
		echo $root/$outputx/$categories
		ls $root/$outputx/$categories/*.pcd > results.list
		python3 render_mitsuba2_pc.py results.list
		rm -rf results.list
		rm -rf $root/$outputx/$catebories/*.exr
		rm -rf $root/$outputx/$catebories/*.xml
	done
done
'
