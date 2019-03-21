# update Weigth

    set ADDR_start_sytolic_array           [expr {0x00100000 + 0x0000}]
    set ADDR_sytolic_array_idle            [expr {0x00100000 + 0x0004}]
    set ADDR_systolic_width_size           [expr {0x00100000 + 0x0008}]
    set ADDR_systolic_height_size          [expr {0x00100000 + 0x000c}]
    set ADDR_input_xin_width_size          [expr {0x00100000 + 0x0010}]
    set ADDR_input_xin_height_size         [expr {0x00100000 + 0x0014}]
    set ADDR_input_acc_size                [expr {0x00100000 + 0x0018}]
    set ADDR_input_channel_start_addr      [expr {0x00100000 + 0x0100}]
    set ADDR_input_channel_shift_ctrl      [expr {0x00100000 + 0x0500}]
    set ADDR_input_acc_channel_start_addr  [expr {0x00100000 + 0x0600}]
    set ADDR_input_acc_channel_shift_ctrl  [expr {0x00100000 + 0x0700}]
    set ADDR_output_acc_channel_start_addr [expr {0x00100000 + 0x0800}]
    set ADDR_output_channel_shift_ctrl     [expr {0x00100000 + 0x0900}]
    set ADDR_column_combine_ctrl           [expr {0x00100000 + 0x0a00}]
    set ADDR_w_control1_ctrl               [expr {0x00100000 + 0x0e00}]
    set ADDR_input_weight_channel_end_addr [expr {0x00100000 + 0x0f00}]

    set ID_WIDTH            2                                                                 
    set SUBARRAY_WIDTH      8                                                                 
    set SUBARRAY_HEIGHT     8                                                                 
    set NUM_DATAFLOW_PER_MX 8                                                                 
    set W_DATA_MUX          3                                   
    set WGT_SRAM_DEPTH      256                                                               
    set WGT_SHIFT_WIDTH     $NUM_DATAFLOW_PER_MX                                               
    set WGT_SRAM_ADDR_W     8                                       
    set ACC_SRAM_DEPTH      [expr {256*256}]                                                           
    set ACC_SRAM_ADDR_W     16                                     
    set XIN_SRAM_DEPTH      [expr {256*256}]                                                           
    set XIN_SRAM_ADDR_W     16                                   
    set N_XIN_PER_MX        8                                                                 
    set N_ACC_PER_MX        32                                                                
    set N_WGT_PER_MX        8                                                                 
    set N_WGT_MX            [expr {$SUBARRAY_WIDTH / $N_WGT_PER_MX}]
    set N_XIN_MX            [expr {$SUBARRAY_WIDTH * $NUM_DATAFLOW_PER_MX/$N_XIN_PER_MX}]
    set N_ACC_MX            [expr {$SUBARRAY_HEIGHT * 4 / $N_ACC_PER_MX}]

    set ADDR_WGT            0x00400000
    set ADDR_XIN            0x00C00000
    set ADDR_DEACC          0x00300000


open_hw
connect_hw_server
open_hw_target


current_hw_device [get_hw_devices xc7vx485t_0]
refresh_hw_device [lindex [get_hw_devices xc7vx485t_0] 0]



##        create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address 0x00010000 -len 1 -size 32 -type read -force
##        run_hw_axi  rd_txn -verbose
##
##
##        create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address 0x00040000 -data 00000001 -len 1 -size 32 -type write -force
##        run_hw_axi  wr_txn -verbose
##
##        create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address 0x00040000 -len 1 -size 32 -type read -force
##        run_hw_axi  rd_txn -verbose

puts "====================================================================="
##        w_rand = 0;
##        for (int i = 0; i < N_WGT_MX; i++) begin
##            for (int j = 0; j < N_WGT_PER_MX; j++) begin
##                for (int k = 0; k < WGT_SRAM_DEPTH; k++) begin
##                    if(k<SUBARRAY_HEIGHT) begin
##                        W[i][j][k] = w_rand;
##                        w_rand++;
##                    end else begin
##                        W[i][j][k] = 32'hf;
##                    end
##                end
##            end
##        end
set w_rand 0
for {set i 0} {$i < $N_WGT_MX} {incr i} {
    for {set j 0} {$j < $N_WGT_PER_MX} {incr j} {
        for {set k 0} {$k < $WGT_SRAM_DEPTH} {incr k} {
            if {$k < $SUBARRAY_HEIGHT} {
                set wr_addr [expr {$ADDR_WGT+4*$k+256*256*16*$j}]
                set data_string [format "0x%08x" $w_rand]
                set addr_string [format "0x%08x" $wr_addr]
                create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
                run_hw_axi  wr_txn -verbose
                #puts "write $wr_addr <= $w_rand"
                puts [format "write %08x <= %01x" $wr_addr $w_rand]
                incr w_rand
            }
        }
    }
}

# setup control1
for {set i $ADDR_w_control1_ctrl} {$i < $ADDR_input_weight_channel_end_addr} {incr i 4} {
    set wr_addr $i
    set data_string [format "0x%08x" 0xffffffff]
    set addr_string [format "0x%08x" $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose
    #puts $addr_string $data_string
}

puts "====================================================================="
# Dump check
for {set i 0} {$i < $N_WGT_MX} {incr i} {
    for {set j 0} {$j < $N_WGT_PER_MX} {incr j} {
        for {set k 0} {$k < $WGT_SRAM_DEPTH} {incr k} {
            if {$k < $SUBARRAY_HEIGHT} {
                set rd_addr     [expr {$ADDR_WGT+4*$k+256*256*16*$j}]
                set addr_string [format "0x%08x" $rd_addr]
                create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
                run_hw_axi  rd_txn -verbose
                set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
                #puts "read $rd_addr => $rdata_tmp"
                scan $rdata_tmp %08x myInteger
                puts [format "read %08x => %d" $rd_addr $myInteger]
            }
        }
    }
}

##        `define XIN_RAM(X) \
##            for (int i = 0; i < XIN_SRAM_DEPTH; i++) begin \
##                g_xin_rd_sram[X].i_xin_sram.RAM[i] = 1; \
##            end
puts "====================================================================="
set w_rand 1
for {set i 0} {$i < 8} {incr i} {
    for {set j 0} {$j < [expr 64*8]} {incr j} {
        set wr_addr [expr {$ADDR_XIN+256*256*16*$i+$j*4}]
        set data_string [format "0x%08x" $w_rand]
        set addr_string [format "0x%08x" $wr_addr]
        create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
        run_hw_axi  wr_txn -verbose
        #puts "write $wr_addr <= $w_rand"
        puts [format "write %08x <= %01x" $wr_addr $w_rand]
    }
}

puts "====================================================================="
for {set i 0} {$i < 8} {incr i} {
    for {set j 0} {$j < 64} {incr j} {
        set rd_addr [expr {$ADDR_XIN+256*256*16*$i+$j*4}]
        set addr_string [format "0x%08x" $rd_addr]
        create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
        run_hw_axi  rd_txn -verbose
        #puts "write $wr_addr <= $w_rand"
        set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
        scan $rdata_tmp %08x myInteger
        puts [format "read %08x => %d" $rd_addr $myInteger]
    }
}



####        // base address is 16 bits, need write 2 cell weight once.
####        for (int i = 0; i < SUBARRAY_WIDTH/2; i++) begin
####            automatic logic [15:0] weight_end_addr = SUBARRAY_HEIGHT-1;
####            issue_apb_tr(.addr(ADDR_input_weight_channel_end_addr+i*4), .rewr(1'b1), .wdata({weight_end_addr, weight_end_addr}), .rdate(rdata));
####        end

puts "====================================================================="
for {set i 0} {$i < [expr $SUBARRAY_WIDTH/2]} {incr i} {
    set weight_end_addr [expr $SUBARRAY_HEIGHT-1]
    set wr_addr         [expr $ADDR_input_weight_channel_end_addr+$i*4]
    set data_string [format "0x%04x%04x" $weight_end_addr $weight_end_addr]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose
    #puts "write $wr_addr <= $w_rand"
    puts [format "write %08x <= %01x" $wr_addr $weight_end_addr]
}

####        for (int i = 0; i < SUBARRAY_WIDTH/2; i++) begin
####            issue_apb_tr(.addr(ADDR_input_weight_channel_end_addr+i*4), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
####        end
####

puts "====================================================================="
####        // update weight size
####        issue_apb_tr(.addr(ADDR_systolic_height_size), .rewr(1'b1), .wdata(SUBARRAY_HEIGHT-1), .rdate(rdata));
####
    set wr_weight_size [expr $SUBARRAY_HEIGHT-1]
    set wr_addr        $ADDR_systolic_height_size
    set data_string [format "0x%08x"     $wr_weight_size]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose

####        // start update weight
####        issue_apb_tr(.addr(ADDR_start_sytolic_array), .rewr(1'b1), .wdata(32'b1), .rdate(rdata));
    set start_weight 1
    set wr_addr        $ADDR_start_sytolic_array
    set data_string [format "0x%08x"     $start_weight]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose

####        //polling status
####        do begin
####            issue_apb_tr(.addr(ADDR_sytolic_array_idle), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
####        end while(rdata[0] != 1'b1);
    set rd_addr        $ADDR_sytolic_array_idle
    set addr_string [format "0x%08x"     $rd_addr]
    create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
    run_hw_axi  rd_txn -verbose
    set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
    scan $rdata_tmp %08x myInteger
    puts [format "read %08x => %d" $rd_addr $myInteger]

####    // update xin size
####    issue_apb_tr(.addr(ADDR_input_xin_width_size ), .rewr(1'b1), .wdata(8-1), .rdate(rdata));
####    issue_apb_tr(.addr(ADDR_input_xin_height_size), .rewr(1'b1), .wdata(8-1), .rdate(rdata));
    set width_size  [expr 8-1]
    set wr_addr        $ADDR_input_xin_width_size
    set data_string [format "0x%08x"     $width_size]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose

    set height_size [expr 8-1]
    set wr_addr        $ADDR_input_xin_height_size
    set data_string [format "0x%08x"     $height_size]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose

####    // update input/output acc size
####    // acc_size = (xin_w * xin_h /4)
####    issue_apb_tr(.addr(ADDR_input_acc_size       ), .rewr(1'b1), .wdata(8*8/4-1), .rdate(rdata));
    set acc_size [expr (8*8/4)-1]
    set wr_addr        $ADDR_input_acc_size
    set data_string [format "0x%08x"     $acc_size]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose

####    // update output channel acc start address
####    for (int i = 0; i < (N_ACC_MX*N_ACC_PER_MX); i=i+2) begin
####        logic [15:0] start_addr0;
####        logic [15:0] start_addr1;
####        start_addr0 =     i*8*8;
####        start_addr1 = (i+1)*8*8;            
####        issue_apb_tr(.addr(ADDR_output_acc_channel_start_addr + 4*i/2), .rewr(1'b1), .wdata({start_addr1, start_addr0}), .rdate(rdata));
####    end
####
for {set i 0} {$i < [expr $N_ACC_MX*$N_ACC_PER_MX]} {incr i 2} {
    set start_addr0 [expr ($i  )*8*8]
    set start_addr1 [expr ($i+1)*8*8]
    set wr_addr     [expr $ADDR_output_acc_channel_start_addr + (4*$i)/2]
    set data_string [format "0x%04x%04x" $start_addr1 $start_addr0]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose
}

### `define DEACC_RAM(X) \
###     for (int i = 0; i < ACC_SRAM_DEPTH; i++) begin \
###         g_deacc_wr_sram[X].i_deacc_sram.RAM[i] = 32'hdeadbeef; \
###     end
### write deacc
set wr_deacc 0xdeadbeef
for {set ch 0} {$ch < 32} {incr ch 1} {
    for {set i 0} {$i < 16} {incr i 1} {
        set wr_addr     [expr $ADDR_DEACC + $ch*64*4 + $i*4]
        set data_string [format "0x%08x"     $wr_deacc]
        set addr_string [format "0x%08x"     $wr_addr]
        create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
        run_hw_axi  wr_txn -verbose
    }
}

####    // start systolic array
####    issue_apb_tr(.addr(ADDR_start_sytolic_array), .rewr(1'b1), .wdata(32'b10), .rdate(rdata));
#for {set i 0} {$i < 100} {incr i 1} {
    set start_mac 0x2
    set wr_addr        $ADDR_start_sytolic_array
    set data_string [format "0x%08x"     $start_mac]
    set addr_string [format "0x%08x"     $wr_addr]
    create_hw_axi_txn wr_txn [get_hw_axis hw_axi_1] -address $addr_string -data $data_string -len 1 -size 32 -type write -force
    run_hw_axi  wr_txn -verbose
#}


#    exit
####    //polling status
####    do begin
####        issue_apb_tr(.addr(ADDR_sytolic_array_idle), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
####    end while(rdata[1] != 1'b1);
for {set i 0} {$i < 1000} {incr i 1} {
    set rd_addr        $ADDR_sytolic_array_idle
    set addr_string [format "0x%08x"     $rd_addr]
    create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
    run_hw_axi  rd_txn -verbose
    set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
    scan $rdata_tmp %08x myInteger
    puts [format "read %08x => %d" $rd_addr $myInteger]
}
# dump deacc
 for {set ch 0} {$ch < 32} {incr ch 1} {
     for {set i 0} {$i < 16} {incr i 1} {
         set rd_addr        [expr $ADDR_DEACC + $ch*64*4 + $i*4]
         set addr_string    [format "0x%08x"     $rd_addr]
         create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
         run_hw_axi  rd_txn -verbose
         set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
         scan $rdata_tmp %08x myInteger
         puts [format "read %08x => %d" $rd_addr $myInteger]
     }
 }




### #for {set ch 0} {$ch < 32} {incr ch 1} {
###     for {set i 0} {$i < 65535} {incr i 1} {
###         set rd_addr        [expr $ADDR_DEACC + $i*4]
###         set addr_string    [format "0x%08x"     $rd_addr]
###         create_hw_axi_txn rd_txn [get_hw_axis hw_axi_1] -address $addr_string -len 1 -size 32 -type read -force
###         run_hw_axi  rd_txn -verbose
###         set rdata_tmp [get_property DATA [get_hw_axi_txn rd_txn]]
###         scan $rdata_tmp %08x myInteger
###         puts [format "read %08x => %d" $rd_addr $myInteger]
###     }
### #}



  
