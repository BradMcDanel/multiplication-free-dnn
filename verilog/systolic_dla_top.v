module systolic_dla_top #(
    parameter           SUBARRAY_WIDTH      = 8                                                ,
    parameter           SUBARRAY_HEIGHT     = 8                                                ,
    parameter           NUM_DATAFLOW_PER_MX = 8                                                ,
    parameter           W_DATA_MUX          = clog2(NUM_DATAFLOW_PER_MX)                       ,
    parameter           WGT_SRAM_DEPTH      = 256                                              ,
    parameter           WGT_SHIFT_WIDTH     = NUM_DATAFLOW_PER_MX                              ,
    parameter           WGT_SRAM_ADDR_W     = clog2(WGT_SRAM_DEPTH)                            ,
    parameter           ACC_SRAM_DEPTH      = 256*256                                          ,
    parameter           ACC_SRAM_ADDR_W     = clog2(ACC_SRAM_DEPTH)                            ,
    parameter           XIN_SRAM_DEPTH      = 256*256                                          ,
    parameter           XIN_SRAM_ADDR_W     = clog2(XIN_SRAM_DEPTH)                            ,
    parameter           N_XIN_PER_MX        = 8                                                ,
    parameter           N_ACC_PER_MX        = 32                                               ,
    parameter           N_WGT_PER_MX        = 8                                                ,
    parameter           N_WGT_MX            = SUBARRAY_WIDTH / N_WGT_PER_MX                    ,
    parameter           N_XIN_MX            = SUBARRAY_WIDTH * NUM_DATAFLOW_PER_MX/N_XIN_PER_MX,
    parameter           N_ACC_MX            = SUBARRAY_HEIGHT * 4 / N_ACC_PER_MX,
    parameter           START_QUANTIZE_BIT  = 11
) (
   input                                        clk             ,
   input                                        reset_n         ,
   // APB interface
   input                                        psel            ,
   input       [                          15:0] paddr           ,
   input                                        pwrite          ,
   input       [                          31:0] pwdata          ,
   input                                        penable         ,
   output      [                          31:0] prdata          ,
   output                                       pready          ,
   // Weight shifter sram interface
   output wire [                8*N_WGT_MX-1:0] wgt_sram_r_en   ,
   output wire [8*WGT_SRAM_ADDR_W*N_WGT_MX-1:0] wgt_sram_raddr  ,
   input  wire [              8*8*N_WGT_MX-1:0] wgt_sram_rdata  ,
   // ACC shifter sram interface
   output wire [                  N_ACC_MX-1:0] acc_sram_r_en   ,
   output wire [  ACC_SRAM_ADDR_W*N_ACC_MX-1:0] acc_sram_raddr  ,
   input  wire [               32*N_ACC_MX-1:0] acc_sram_rdata  ,
   // ACC deshifter sram interface
   output wire [                  N_ACC_MX-1:0] deacc_sram_w_en ,
   output wire [  ACC_SRAM_ADDR_W*N_ACC_MX-1:0] deacc_sram_waddr,
   output wire [               32*N_ACC_MX-1:0] deacc_sram_wdata,
   // XIN sram interface
   output wire [                  N_XIN_MX-1:0] xin_sram_r_en   ,
   output wire [  XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_sram_raddr  ,
   input  wire [                8*N_XIN_MX-1:0] xin_sram_rdata
); 
    genvar i, j;

    function [31:0] clog2 (input [31:0] x);
        reg [31:0] x_tmp;
        begin
                x_tmp = x-1;
                for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                    x_tmp = x_tmp >> 1;
                end
        end
    endfunction

    wire [31:0] reg_write_data;
    wire [15:0] reg_addr      ;
    wire [31:0] reg_read_data ;
    wire        reg_write     ;
    wire        reg_read      ;
    wire        reg_idle      ;
    
    // access the reg file with the axi2apb interface
    
    apb2reg i_apb2reg (
        .clk           (clk           ),
        .reset_n       (reset_n       ),
        .psel          (psel          ),
        .paddr         (paddr[15:2]   ),
        .pwrite        (pwrite        ),
        .pwdata        (pwdata        ),
        .penable       (penable       ),
        .prdata        (prdata        ),
        .pready        (pready        ),
        .reg_write_data(reg_write_data),
        .reg_addr      (reg_addr      ),
        .reg_read_data (reg_read_data ),
        .reg_write     (reg_write     ),
        .reg_read      (reg_read      ),
        .reg_idle      (reg_idle      )
    );

    wire [   1:0] start_sytolic_array          ;
    wire [   1:0] sytolic_array_idle           ;
    wire [8191:0] input_channel_start_addr     ;
    wire [2047:0] input_channel_shift_ctrl     ;
    wire [2047:0] input_acc_channel_start_addr ;
    wire [ 127:0] input_acc_channel_shift_ctrl ;
    wire [2047:0] output_acc_channel_start_addr;
    wire          output_channel_shift_ctrl    ;
    wire [8191:0] column_combine_ctrl          ;
    wire [2047:0] w_control1_ctrl              ;
    wire [1023:0] input_weight_channel_end_addr;
    wire [  31:0] systolic_width_size          ;
    wire [  31:0] systolic_height_size         ;
    wire [  31:0] input_xin_width_size         ;
    wire [  31:0] input_xin_height_size        ;
    wire [  31:0] input_acc_size               ;
    
    // save the signal to the reg file
    reg_define i_reg_define (
        .start_sytolic_array          (start_sytolic_array          ),
        .sytolic_array_idle           (sytolic_array_idle           ),
        .systolic_width_size          (systolic_width_size          ),
        .systolic_height_size         (systolic_height_size         ),
        .input_xin_width_size         (input_xin_width_size         ),
        .input_xin_height_size        (input_xin_height_size        ),
        .input_acc_size               (input_acc_size               ),
        .input_channel_start_addr     (input_channel_start_addr     ),
        .input_channel_shift_ctrl     (input_channel_shift_ctrl     ),
        .input_acc_channel_start_addr (input_acc_channel_start_addr ),
        .input_acc_channel_shift_ctrl (input_acc_channel_shift_ctrl ),
        .output_acc_channel_start_addr(output_acc_channel_start_addr),
        .output_channel_shift_ctrl    (output_channel_shift_ctrl    ),
        .column_combine_ctrl          (column_combine_ctrl          ),
        .input_weight_channel_end_addr(input_weight_channel_end_addr),
        .w_control1_ctrl              (w_control1_ctrl              ),
        .write_data                   (reg_write_data               ),
        .addr                         (reg_addr                     ),
        .read_data                    (reg_read_data                ),
        .write                        (reg_write                    ),
        .read                         (reg_read                     ),
        .clk                          (clk                          ),
        .reset_l                      (reset_n                      )
    );

    // Activation input or feature map input
    wire [                  N_XIN_MX-1:0] xin_shift_start    ;
    wire [                  N_XIN_MX-1:0] xin_shift_start_o  ;
    wire [                  N_XIN_MX-1:0] xin_shift_idle     ;
    wire [              8*4*N_XIN_MX-1:0] xin_shift_ctrl     ;
    wire [8*XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_start_addr     ;
    wire [           XIN_SRAM_ADDR_W-1:0] xin_img_width_size ;
    wire [           XIN_SRAM_ADDR_W-1:0] xin_img_height_size;
    wire [                8*N_XIN_MX-1:0] xin_serial_output  ;
    wire [                8*N_XIN_MX-1:0] xin_serial_en      ;
    wire [ SUBARRAY_WIDTH-1:0] is_zero_mux;
    wire [                8*N_XIN_MX-1:0] is_zero_output  ;  

    generate
        for (i = 0; i < N_XIN_MX; i=i+1) begin: g_xin_shifter
            if(i==0) begin 
                j_shifter_MX_cell #(.SRAM_DEPTH(XIN_SRAM_DEPTH),.GENERATE_MUX_SINGAL(1)) i_j_shifter_MX_cell (
                    .clk            (clk                                                             ),
                    .reset_n        (reset_n                                                         ),
                    .sram_en        (xin_sram_r_en        [i]                                        ),
                    .sram_addr      (xin_sram_raddr       [i*XIN_SRAM_ADDR_W   +: XIN_SRAM_ADDR_W  ] ),
                    .sram_data      ({1'b0, xin_sram_rdata[i*8                 +: 8                ]}),
                    .shift_start    (xin_shift_start      [i]                                        ),
                    .shift_start_o  (xin_shift_start_o    [i]                                        ),
                    .shift_idle     (xin_shift_idle       [i]                                        ),
                    .shift_ctrl     (xin_shift_ctrl       [i*4*8               +: 4*8              ] ),
                    .start_addr     (xin_start_addr       [i*XIN_SRAM_ADDR_W*8 +: XIN_SRAM_ADDR_W*8] ),
                    .img_width_size (xin_img_width_size                                              ),
                    .img_height_size(xin_img_height_size                                             ),
                    .is_zero_mux    (is_zero_mux),
                    .is_zero_output (is_zero_output[i*8                 +: 8                ]),
                    .serial_output  (xin_serial_output    [i*8                 +: 8                ] ),
                    .serial_en      (xin_serial_en        [i*8                 +: 8                ] )
                );
                assign xin_shift_start[i] = start_sytolic_array[1];
            end
            else begin
                j_shifter_MX_cell #(.SRAM_DEPTH(XIN_SRAM_DEPTH),.GENERATE_MUX_SINGAL(0)) i_j_shifter_MX_cell (
                    .clk            (clk                                                             ),
                    .reset_n        (reset_n                                                         ),
                    .sram_en        (xin_sram_r_en        [i]                                        ),
                    .sram_addr      (xin_sram_raddr       [i*XIN_SRAM_ADDR_W   +: XIN_SRAM_ADDR_W  ] ),
                    .sram_data      ({1'b0, xin_sram_rdata[i*8                 +: 8                ]}),
                    .shift_start    (xin_shift_start      [i]                                        ),
                    .shift_start_o  (xin_shift_start_o    [i]                                        ),
                    .shift_idle     (xin_shift_idle       [i]                                        ),
                    .shift_ctrl     (xin_shift_ctrl       [i*4*8               +: 4*8              ] ),
                    .start_addr     (xin_start_addr       [i*XIN_SRAM_ADDR_W*8 +: XIN_SRAM_ADDR_W*8] ),
                    .img_width_size (xin_img_width_size                                              ),
                    .img_height_size(xin_img_height_size                                             ),
                    .is_zero_output (is_zero_output[i*8                 +: 8                ]),
                    .serial_output  (xin_serial_output    [i*8                 +: 8                ] ),
                    .serial_en      (xin_serial_en        [i*8                 +: 8                ] )
                );
                //assign xin_shift_start[i] = xin_shift_start_o[i-1];
                // there is a problem here, will change
                assign xin_shift_start[i] = start_sytolic_array[1];
            end
        end

        for (i = 0; i < 8*N_XIN_MX; i=i+1) begin
            assign xin_start_addr       [i*XIN_SRAM_ADDR_W +: XIN_SRAM_ADDR_W] = 
                input_channel_start_addr[i*16              +: XIN_SRAM_ADDR_W];
            assign xin_shift_ctrl       [i*4               +: 4] =
                input_channel_shift_ctrl[i*4               +: 4];
        end
    endgenerate

    assign xin_img_width_size = input_xin_width_size [XIN_SRAM_ADDR_W-1:0];
    assign xin_img_height_size= input_xin_height_size[XIN_SRAM_ADDR_W-1:0];

    // Weight input
    wire [8*WGT_SRAM_ADDR_W*N_WGT_MX-1:0] wgt_end_addr     ;
    wire [8*WGT_SHIFT_WIDTH*N_WGT_MX-1:0] wgt_serial_output;
    wire [                8*N_WGT_MX-1:0] wgt_serial_en    ;
    wire [           WGT_SRAM_ADDR_W-1:0] wgt_img_size     ;
    wire [                  N_WGT_MX-1:0] wgt_shift_start  ;
    wire [                  N_WGT_MX-1:0] wgt_shift_idle   ;

   generate
        for (i = 0; i < N_WGT_MX; i=i+1) begin: g_wgt_shifter
            j_wgt_shifter_MX_cell #(
                .SRAM_DEPTH (WGT_SRAM_DEPTH ),
                .SHIFT_WIDTH(WGT_SHIFT_WIDTH)
            ) i_j_wgt_shifter_MX_cell (
                .clk          (clk                                                         ),
                .reset_n      (reset_n                                                     ),
                .sram_en      (wgt_sram_r_en     [i*8                 +: 8                ]),
                .sram_addr    (wgt_sram_raddr    [i*8*WGT_SRAM_ADDR_W +: 8*WGT_SRAM_ADDR_W]),
                .sram_data    (wgt_sram_rdata    [i*8*8               +: 8*8              ]),
                .shift_start  (wgt_shift_start   [i]                                       ),
                .shift_idle   (wgt_shift_idle    [i]                                       ),
                .end_addr     (wgt_end_addr      [i*8*WGT_SRAM_ADDR_W +: 8*WGT_SRAM_ADDR_W]),
                .img_size     (wgt_img_size                                                ),
                .serial_output(wgt_serial_output [i*8*WGT_SHIFT_WIDTH +: 8*WGT_SHIFT_WIDTH]),
                .serial_en    (wgt_serial_en     [i*8                 +: 8                ])
            );
    
            assign wgt_shift_start[i] = start_sytolic_array[0];
        end

        for (i = 0; i < N_WGT_MX*8 ; i=i+1) begin
            assign wgt_end_addr                  [i*WGT_SRAM_ADDR_W +: WGT_SRAM_ADDR_W]=
                   input_weight_channel_end_addr [i*16              +: WGT_SRAM_ADDR_W];
        end
    endgenerate

    assign wgt_img_size = systolic_height_size[WGT_SRAM_ADDR_W-1:0];
    assign sytolic_array_idle[0] = (&wgt_shift_idle);

    // Accumulator input: Zero or read sram
    wire [                   N_ACC_MX-1:0] acc_shift_start  ;
    wire [                   N_ACC_MX-1:0] acc_shift_start_o;
    wire [                   N_ACC_MX-1:0] acc_shift_idle   ;
    wire [                32*N_ACC_MX-1:0] acc_shift_ctrl   ;
    wire [32*ACC_SRAM_ADDR_W*N_ACC_MX-1:0] acc_start_addr   ;
    wire [            ACC_SRAM_ADDR_W-1:0] acc_img_size     ;
    wire [                32*N_ACC_MX-1:0] acc_serial_output;
    wire [                32*N_ACC_MX-1:0] acc_serial_en    ;
    wire [                32*N_ACC_MX-1:0] acc_serial_start ;

   generate
        for (i = 0; i < N_ACC_MX; i=i+1) begin: g_acc_shifter
            j_acc_shifter_MX_cell #(.SRAM_DEPTH(ACC_SRAM_DEPTH)) i_j_acc_shifter_MX_cell (
                .clk          (clk                                                            ),
                .reset_n      (reset_n                                                        ),
                .sram_en      (acc_sram_r_en    [i]                                           ),
                .sram_addr    (acc_sram_raddr   [i*ACC_SRAM_ADDR_W      +: ACC_SRAM_ADDR_W   ]),
                .sram_data    (acc_sram_rdata   [i*32                   +: 32                ]),
                .shift_start  (acc_shift_start  [i]                                           ),
                .shift_start_o(acc_shift_start_o[i]                                           ),
                .shift_idle   (acc_shift_idle   [i]                                           ),
                .shift_ctrl   (acc_shift_ctrl   [i*32                   +: 32                ]),
                .start_addr   (acc_start_addr   [i*ACC_SRAM_ADDR_W*32   +: ACC_SRAM_ADDR_W*32]),
                .img_size     (acc_img_size                                                   ),
                .serial_output(acc_serial_output[i*32                   +: 32                ]),
                .serial_en    (acc_serial_en    [i*32                   +: 32                ]),
                .serial_start (acc_serial_start [i*32                   +: 32                ])
            );

            if(i==0) begin
                assign acc_shift_start[i] = start_sytolic_array[1];
            end else begin
                assign acc_shift_start[i] = acc_shift_start_o[i-1];
            end
        end

        for (i = 0; i < 32*N_ACC_MX; i=i+1) begin
            assign acc_start_addr               [i*ACC_SRAM_ADDR_W +: ACC_SRAM_ADDR_W] =
                   input_acc_channel_start_addr [i*16              +: ACC_SRAM_ADDR_W];
            // Careful this, the define of shift control is inversed.
            assign acc_shift_ctrl               [i] =
                  ~input_acc_channel_shift_ctrl [i];
        end
    endgenerate

    assign acc_img_size = input_acc_size[ACC_SRAM_ADDR_W-1:0];


    // Write to Deshifter
    wire [                   N_ACC_MX-1:0] deacc_shift_start  ;
    wire [                   N_ACC_MX-1:0] deacc_shift_start_o;
    wire [                   N_ACC_MX-1:0] deacc_shift_idle   ;
    wire [32*ACC_SRAM_ADDR_W*N_ACC_MX-1:0] deacc_start_addr   ;
    wire [            ACC_SRAM_ADDR_W-1:0] deacc_img_size     ;
    wire [                32*N_ACC_MX-1:0] deacc_serial_input ;
    wire [                32*N_ACC_MX-1:0] deacc_serial_en    ;

    
    generate
        for (i = 0; i < N_ACC_MX; i=i+1) begin: g_acc_deshifter
            j_acc_deshifter_MX_cell #(.SRAM_DEPTH(ACC_SRAM_DEPTH)) i_j_acc_deshifter_MX_cell (
                .clk          (clk                                                              ),
                .reset_n      (reset_n                                                          ),
                .sram_en      (deacc_sram_w_en    [i]                                           ),
                .sram_addr    (deacc_sram_waddr   [i*ACC_SRAM_ADDR_W      +: ACC_SRAM_ADDR_W   ]),
                .sram_data    (deacc_sram_wdata   [i*32                   +: 32                ]),
                .shift_start  (deacc_shift_start  [i]                                           ),
                .shift_start_o(deacc_shift_start_o[i]                                           ),
                .shift_idle   (deacc_shift_idle   [i]                                           ),
                .start_addr   (deacc_start_addr   [i*ACC_SRAM_ADDR_W*32   +: ACC_SRAM_ADDR_W*32]),
                .img_size     (deacc_img_size                                                   ),
                .serial_input (deacc_serial_input [i*32                   +: 32                ]),
                .serial_en    (deacc_serial_en    [i*32                   +: 32                ])
            );
            if(i==0) begin
                assign deacc_shift_start[i] = start_sytolic_array[1];
            end else begin
                assign deacc_shift_start[i] = deacc_shift_start_o[i-1];
            end
        end

        for (i = 0; i < 32*N_ACC_MX; i=i+1) begin
            assign deacc_start_addr              [i*ACC_SRAM_ADDR_W +: ACC_SRAM_ADDR_W] =
                   output_acc_channel_start_addr [i*16              +: ACC_SRAM_ADDR_W];
        end
    endgenerate

    assign deacc_img_size = input_acc_size[ACC_SRAM_ADDR_W-1:0];
    
    
    

    // Systolic main 
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_accumulation_in ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_clr_and_plus_one;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_serial_end      ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_mac_en          ;
    wire [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_wgt ;
    wire [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_xin ;
    wire [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] systolic_dataflow_in     ;
    wire [SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX-1:0] systolic_dataflow_select ;
    wire [                           SUBARRAY_WIDTH-1:0] systolic_update_w        ;
    wire [           SUBARRAY_WIDTH*SUBARRAY_HEIGHT-1:0] systolic_control1        ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_result          ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_result_en       ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_result_start    ;
    wire [                        4*SUBARRAY_HEIGHT-1:0] systolic_result_end      ;
    wire [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] zero_dataflow_in     ;
    wire [       SUBARRAY_WIDTH-1:0] zero_sel_in     ;
    j_systolic_array #(
        .SUBARRAY_WIDTH     (SUBARRAY_WIDTH ),
        .SUBARRAY_HEIGHT    (SUBARRAY_HEIGHT),
        .NUM_DATAFLOW_PER_MX(8              )
    ) i_j_systolic_array (
        .clk             (clk                      ),
        .reset           (~reset_n                 ),
        .accumulation_in (systolic_accumulation_in ),
        .clr_and_plus_one(systolic_clr_and_plus_one),
        .serial_end      (systolic_serial_end      ),
        .mac_en          (systolic_mac_en          ),
        .dataflow_in     (systolic_dataflow_in     ),
        .dataflow_select (systolic_dataflow_select ),
        .update_w        (systolic_update_w        ),
        .control1        (systolic_control1        ),
        .zero_dataflow_in(zero_dataflow_in),
        .zero_sel_in     (zero_sel_in),
        .result          (systolic_result          ),
        .result_en       (systolic_result_en       ),
        .result_start    (systolic_result_start    ),
        .result_end      (systolic_result_end      )
    );

    assign systolic_serial_end = {(4*SUBARRAY_HEIGHT){1'b0}};

    generate
        for (i = 0; i < ((SUBARRAY_HEIGHT*4)/N_ACC_PER_MX); i=i+1) 
        begin: g_h_connect
            for (j = 0; j < (SUBARRAY_HEIGHT*4); j=j+1)
                begin
                    // Be careful the acc index is increase, but systolic array need jump 8
                    // Accumulator
                    assign systolic_clr_and_plus_one [i*N_ACC_PER_MX*32+(j%8)*4+(j/8)] = acc_serial_start   [i*N_ACC_PER_MX*32+j];
                    assign systolic_mac_en           [i*N_ACC_PER_MX*32+(j%8)*4+(j/8)] = acc_serial_en      [i*N_ACC_PER_MX*32+j];
                    assign systolic_accumulation_in  [i*N_ACC_PER_MX*32+(j%8)*4+(j/8)] = acc_serial_output  [i*N_ACC_PER_MX*32+j];
                    // De-Accumulator
                    assign deacc_serial_input [i*N_ACC_PER_MX*32+j] = systolic_result    [i*N_ACC_PER_MX*32+(j%8)*4+(j/8)];     // when connect to the systolic array, it must be in order
                    assign deacc_serial_en    [i*N_ACC_PER_MX*32+j] = systolic_result_en [i*N_ACC_PER_MX*32+(j%8)*4+(j/8)];
                end
        end

        for (i = 0; i < SUBARRAY_WIDTH; i=i+1) begin: g_column_ctrl
            for (j = 0; j < SUBARRAY_HEIGHT; j=j+1) begin
                assign systolic_dataflow_select[(i*SUBARRAY_HEIGHT+j)*3 +: 3]  = column_combine_ctrl[(i*SUBARRAY_HEIGHT+j)*4 +: 3];
                // 0 for postive value, 1 for negative value
                assign systolic_control1       [i*SUBARRAY_WIDTH+j] =~w_control1_ctrl[i*SUBARRAY_WIDTH+j];
            end
        end

        // Update weigth
        for (j = 0; j < SUBARRAY_WIDTH; j=j+1) begin
            assign systolic_update_w   [j] = wgt_serial_en[j];
            for (i = 0; i < NUM_DATAFLOW_PER_MX; i=i+1) begin: g_weight_ctrl
                assign systolic_dataflow_in_wgt[j*NUM_DATAFLOW_PER_MX + i] =
                            systolic_update_w[j] ? wgt_serial_output[j*NUM_DATAFLOW_PER_MX + i]: 1'b0;     
            end
        end

        // Activate
        for (j = 0; j < SUBARRAY_WIDTH; j=j+1) begin
            assign zero_sel_in[j] = ~systolic_update_w[j] ? is_zero_mux[j]: 1'b0;                         
            for (i = 0; i < NUM_DATAFLOW_PER_MX; i=i+1) begin: g_activate
                // The cross bar wire connect must careful
                assign systolic_dataflow_in_xin[j*NUM_DATAFLOW_PER_MX + i] =
                            ~systolic_update_w[j] ? xin_serial_output[i*NUM_DATAFLOW_PER_MX + j]: 1'b0;    
                assign zero_dataflow_in[j*NUM_DATAFLOW_PER_MX + i] =
                            ~systolic_update_w[j] ? is_zero_output[i*NUM_DATAFLOW_PER_MX + j]: 1'b0;   
            end
        end

    endgenerate

    assign systolic_dataflow_in = systolic_dataflow_in_wgt | systolic_dataflow_in_xin;


    // idle
    assign sytolic_array_idle[1] = (&deacc_shift_idle) & (&acc_shift_idle) & (&xin_shift_idle);

endmodule


