import impkp_core
import viewer

## BASIC USER PARAMETERS ##
project_name = '20220323_5cm_38RHdig_speeds_0p4_B01_run1'

y_min = 2
y_max = 15

x_min = 35
x_max = 88
## END OF BASIC USER PARAMETERS ##

# for best results, position probe at (x_min, y_min) -- 
# that is, to the lower left corner on TOGUARD TV

## ADVANCED PARAMETERS ##
total_lines_in_x_axis = 200
lines_per_fragment = 10
fast_stage_speed = 0.2 # mm/s
## END OF ADVANCED PARAMETERS ##

impkp_core.scan_large_area([y_min, x_min], [y_max, x_max], 
                           total_lines_in_x_axis, 
                           4000, 
                           lines_per_fragment, 
                           fast_stage_speed, 
                           project_name)
viewer.postprocess_folder(project_name, 
                   bkg_pos='left',
                   force_zero_shifts = False,
                   )