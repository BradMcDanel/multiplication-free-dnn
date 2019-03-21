module tb_systolic_dla_top #(
    parameter SUBARRAY_WIDTH      = 8                                                ,
    parameter SUBARRAY_HEIGHT     = 8                                                ,
    parameter NUM_DATAFLOW_PER_MX = 8                                                ,
    parameter W_DATA_MUX          = clog2(NUM_DATAFLOW_PER_MX)                       ,
    parameter WGT_SRAM_DEPTH      = 256                                              ,
    parameter WGT_SHIFT_WIDTH     = NUM_DATAFLOW_PER_MX                              ,
    parameter WGT_SRAM_ADDR_W     = clog2(WGT_SRAM_DEPTH)                            ,
    parameter ACC_SRAM_DEPTH      = 256*256                                          ,
    parameter ACC_SRAM_ADDR_W     = clog2(ACC_SRAM_DEPTH)                            ,
    parameter XIN_SRAM_DEPTH      = 256*256                                          ,
    parameter XIN_SRAM_ADDR_W     = clog2(XIN_SRAM_DEPTH)                            ,
    parameter N_XIN_PER_MX        = 8                                                ,
    parameter N_ACC_PER_MX        = 32                                               ,
    parameter N_WGT_PER_MX        = 8                                                ,
    parameter N_WGT_MX            = SUBARRAY_WIDTH / N_WGT_PER_MX                    ,
    parameter N_XIN_MX            = SUBARRAY_WIDTH * NUM_DATAFLOW_PER_MX/N_XIN_PER_MX,
    parameter N_ACC_MX            = SUBARRAY_HEIGHT * 4 / N_ACC_PER_MX
)(); 

    `include "reg_define.vh"

    logic clk    ;
    logic reset_n;
    // APB interface
    logic        psel    = 'b0;
    logic [15:0] paddr   = 'b0;
    logic        pwrite  = 'b0;
    logic [31:0] pwdata  = 'b0;
    logic        penable = 'b0;
    wire  [31:0] prdata ;
    wire         pready ;
    // Weight shifter sram interface
    wire [                8*N_WGT_MX-1:0] wgt_sram_r_en ;
    wire [8*WGT_SRAM_ADDR_W*N_WGT_MX-1:0] wgt_sram_raddr;
    wire [              8*8*N_WGT_MX-1:0] wgt_sram_rdata;
    // ACC shifter sram interface
    wire [                N_ACC_MX-1:0] acc_sram_r_en ;
    wire [ACC_SRAM_ADDR_W*N_ACC_MX-1:0] acc_sram_raddr;
    wire [             32*N_ACC_MX-1:0] acc_sram_rdata;
    // ACC deshifter sram interface
    wire [                N_ACC_MX-1:0] deacc_sram_w_en ;
    wire [ACC_SRAM_ADDR_W*N_ACC_MX-1:0] deacc_sram_waddr;
    wire [             32*N_ACC_MX-1:0] deacc_sram_wdata;
    // XIN sram interface
    wire [                N_XIN_MX-1:0] xin_sram_r_en ;
    wire [XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_sram_raddr;
    wire [              8*N_XIN_MX-1:0] xin_sram_rdata;

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


    generate
        for (i = 0; i < N_WGT_MX; i=i+1) begin: g_wgt_rd_sram_i
            for (j = 0; j < 8; j=j+1) begin: g_wgt_rd_sram_j
                sram #(
                    .DATA_WIDTH(8              ),
                    .RAM_SIZE  (WGT_SRAM_DEPTH ),
                    .ADDR_WIDTH(WGT_SRAM_ADDR_W)
                ) i_wgt_sram (
                    .clk   (clk                                                       ),
                    .we    (1'b0                                                      ),
                    .en    (wgt_sram_r_en [(i*8+j)                                   ]),
                    .addr  (wgt_sram_raddr[(i*8+j)*WGT_SRAM_ADDR_W +: WGT_SRAM_ADDR_W]),
                    .data_i(8'b0                                                      ),
                    .data_o(wgt_sram_rdata[(i*8+j)*8               +: 8              ])
                );                
            end
        end
    endgenerate

    generate
        for (i = 0; i < N_XIN_MX; i=i+1) begin: g_xin_rd_sram
            sram #(
                .DATA_WIDTH(8              ),
                .RAM_SIZE  (XIN_SRAM_DEPTH ),
                .ADDR_WIDTH(XIN_SRAM_ADDR_W)
            ) i_xin_sram (
                .clk   (clk                                                 ),
                .we    (1'b0                                                ),
                .en    (xin_sram_r_en [i]                                   ),
                .addr  (xin_sram_raddr[i*XIN_SRAM_ADDR_W +: XIN_SRAM_ADDR_W]),
                .data_i(8'b0                                                ),
                .data_o(xin_sram_rdata[i*8               +: 8]              )
            );
        end
    endgenerate

    generate
        for (i = 0; i < N_ACC_MX; i=i+1) begin: g_acc_rd_sram
            sram #(
                .DATA_WIDTH(32             ),
                .RAM_SIZE  (ACC_SRAM_DEPTH ),
                .ADDR_WIDTH(ACC_SRAM_ADDR_W)
            ) i_acc_sram (
                .clk   (clk                                                 ),
                .we    (1'b0                                                ),
                .en    (acc_sram_r_en [i]                                   ),
                .addr  (acc_sram_raddr[i*ACC_SRAM_ADDR_W +: ACC_SRAM_ADDR_W]),
                .data_i(32'b0                                               ),
                .data_o(acc_sram_rdata[i*32              +: 32]             )
            );
        end
    endgenerate

    generate
        for (i = 0; i < N_ACC_MX; i=i+1) begin: g_deacc_wr_sram
            sram #(
                .DATA_WIDTH(32             ),
                .RAM_SIZE  (ACC_SRAM_DEPTH ),
                .ADDR_WIDTH(ACC_SRAM_ADDR_W)
            ) i_deacc_sram (
                .clk   (clk                                                   ),
                .we    (deacc_sram_w_en [i]                                   ),
                .en    (deacc_sram_w_en [i]                                   ),
                .addr  (deacc_sram_waddr[i*ACC_SRAM_ADDR_W +: ACC_SRAM_ADDR_W]),
                .data_i(deacc_sram_wdata[i*32              +: 32]             ),
                .data_o(                                                      )
            );
        end
    endgenerate

    systolic_dla_top #(
        .SUBARRAY_WIDTH     (SUBARRAY_WIDTH     ),
        .SUBARRAY_HEIGHT    (SUBARRAY_WIDTH     ),
        .NUM_DATAFLOW_PER_MX(NUM_DATAFLOW_PER_MX),
        .WGT_SRAM_DEPTH     (WGT_SRAM_DEPTH     ),
        .ACC_SRAM_DEPTH     (ACC_SRAM_DEPTH     ),
        .XIN_SRAM_DEPTH     (XIN_SRAM_DEPTH     )
    ) i_systolic_dla_top (
        .clk             (clk             ),
        .reset_n         (reset_n         ),
        .psel            (psel            ),
        .paddr           (paddr           ),
        .pwrite          (pwrite          ),
        .pwdata          (pwdata          ),
        .penable         (penable         ),
        .prdata          (prdata          ),
        .pready          (pready          ),
        .wgt_sram_r_en   (wgt_sram_r_en   ),
        .wgt_sram_raddr  (wgt_sram_raddr  ),
        .wgt_sram_rdata  (wgt_sram_rdata  ),
        .acc_sram_r_en   (acc_sram_r_en   ),
        .acc_sram_raddr  (acc_sram_raddr  ),
        .acc_sram_rdata  (acc_sram_rdata  ),
        .deacc_sram_w_en (deacc_sram_w_en ),
        .deacc_sram_waddr(deacc_sram_waddr),
        .deacc_sram_wdata(deacc_sram_wdata),
        .xin_sram_r_en   (xin_sram_r_en   ),
        .xin_sram_raddr  (xin_sram_raddr  ),
        .xin_sram_rdata  (xin_sram_rdata  )
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        reset_n = 1'b1;
        @(posedge clk);
        reset_n = 1'b0;
        @(posedge clk);
        reset_n = 1'b1;
    end

    initial begin
      $dumpfile("test.vcd");
      $dumpvars;
    end

    task automatic issue_apb_tr(
            input   [15:0] addr,
            input          rewr,
            input   [31:0] wdata,
            output  [31:0] rdate
        );
        psel   = 1'b1;
        paddr  = addr;
        pwrite = rewr;
        pwdata = rewr ? wdata : 32'bz;

        @(posedge clk);
        penable = 1'b1;

        @(posedge clk);
        while(~pready) @(posedge clk);

        rdate = ~rewr ? prdata : 32'bz;
        psel     = 1'b0;
        penable  = 1'b0;

        paddr    = 32'bz;
        pwrite   = 1'b0;
        pwdata   = 32'bz;
        if(rewr) $display($stime, " APB WR [%h] <= %h", addr, wdata);
        else     $display($stime, " APB RE [%h] <= %h", addr, rdate);
    endtask

    task automatic get_clk();
        @(posedge clk);
    endtask : get_clk


    task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
        if(golden !== check_result) begin
            $error($stime, " Error golden %d != got %d", golden, check_result);
            $finish;
        end
    endtask : check_32

    task automatic check_8(input logic [7:0] golden, input logic [7:0] check_result, input string err_msg="");
        if(golden !== check_result) begin
            $error($stime, " Error golden %d != got %d", golden, check_result);
            $finish;
        end
    endtask : check_8


    int W[N_WGT_MX+1][N_WGT_PER_MX][WGT_SRAM_DEPTH];

    initial begin
        logic [31:0] rdata;
        int w_rand;

        repeat(10) get_clk;

        w_rand = 128;
        for (int i = 0; i < N_WGT_MX; i++) begin
            for (int j = 0; j < N_WGT_PER_MX; j++) begin
                for (int k = 0; k < WGT_SRAM_DEPTH; k++) begin
                    if(k<SUBARRAY_HEIGHT) begin
                        W[i][j][k] = w_rand;
                        w_rand++;
                    end else begin
                        W[i][j][k] = 32'hf;
                    end
                end
            end
        end

        `define WGT_RAM(X, Y) \
            for (int i = 0; i < WGT_SRAM_DEPTH; i++) begin \
                g_wgt_rd_sram_i[X].g_wgt_rd_sram_j[Y].i_wgt_sram.RAM[i] = W[X][Y][i]; \
            end

        `define XIN_RAM(X) \
            for (int i = 0; i < XIN_SRAM_DEPTH; i++) begin \
                if (i%2 == 0) begin \
                    g_xin_rd_sram[X].i_xin_sram.RAM[i] = 33; \
                end \
                if (i%2 != 0) begin \
                    g_xin_rd_sram[X].i_xin_sram.RAM[i] = 33; \
                end \
            end

        `define ACC_RAM(X) \
            for (int i = 0; i < ACC_SRAM_DEPTH; i++) begin \
                g_acc_rd_sram[X].i_acc_sram.RAM[i] = 1; \
            end

        `define DEACC_RAM(X) \
            for (int i = 0; i < ACC_SRAM_DEPTH; i++) begin \
                g_deacc_wr_sram[X].i_deacc_sram.RAM[i] = 32'hdeadbeef; \
            end
        `define READ_DEACC_RAM(X) \
           for (int i = 0; i < ACC_SRAM_DEPTH; i++) begin \
               $display($stime, " Item in addr [%d] is [%h]", i , g_deacc_wr_sram[X].i_deacc_sram.RAM[i]); \
           end
        
        `WGT_RAM(00, 00)
        `WGT_RAM(00, 01)
        `WGT_RAM(00, 02)
        `WGT_RAM(00, 03)
        `WGT_RAM(00, 04)
        `WGT_RAM(00, 05)
        `WGT_RAM(00, 06)
        `WGT_RAM(00, 07)

        `XIN_RAM(00)
        `XIN_RAM(01)
        `XIN_RAM(02)
        `XIN_RAM(03)
        `XIN_RAM(04)
        `XIN_RAM(05)
        `XIN_RAM(06)
        `XIN_RAM(07)

        `ACC_RAM(00)
        `DEACC_RAM(00)


        // base address is 16 bits, need write 2 cell weight once.
        for (int i = 0; i < SUBARRAY_WIDTH/2; i++) begin
            automatic logic [15:0] weight_end_addr = SUBARRAY_HEIGHT-1;
            issue_apb_tr(.addr(ADDR_input_weight_channel_end_addr+i*4), .rewr(1'b1), .wdata({weight_end_addr, weight_end_addr}), .rdate(rdata));
        end

        for (int i = 0; i < SUBARRAY_WIDTH/2; i++) begin
            issue_apb_tr(.addr(ADDR_input_weight_channel_end_addr+i*4), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
        end

        // update weight size
        issue_apb_tr(.addr(ADDR_systolic_height_size), .rewr(1'b1), .wdata(SUBARRAY_HEIGHT-1), .rdate(rdata));

        // start update weight
        issue_apb_tr(.addr(ADDR_start_sytolic_array), .rewr(1'b1), .wdata(32'b1), .rdate(rdata));

        //polling status
        do begin
            issue_apb_tr(.addr(ADDR_sytolic_array_idle), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
        end while(rdata[0] != 1'b1);


        `define CHECK_WGT(H_idx, W_idx) \
            $display($stime, " Check W[%-02d][%-02d] = 0x%-02h, 0x%-02h", W_idx, H_idx, W[0][W_idx][H_idx], \
                                    i_systolic_dla_top.i_j_systolic_array.g_H[H_idx].g_W[W_idx].i_j_MX_cell.shared_W); \
            check_8(W[0][W_idx][H_idx], i_systolic_dla_top.i_j_systolic_array.g_H[H_idx].g_W[W_idx].i_j_MX_cell.shared_W);

        `CHECK_WGT(0,0) `CHECK_WGT(0,1) `CHECK_WGT(0,2) `CHECK_WGT(0,3) `CHECK_WGT(0,4) `CHECK_WGT(0,5) `CHECK_WGT(0,6) `CHECK_WGT(0,7)
        `CHECK_WGT(1,0) `CHECK_WGT(1,1) `CHECK_WGT(1,2) `CHECK_WGT(1,3) `CHECK_WGT(1,4) `CHECK_WGT(1,5) `CHECK_WGT(1,6) `CHECK_WGT(1,7)
        `CHECK_WGT(2,0) `CHECK_WGT(2,1) `CHECK_WGT(2,2) `CHECK_WGT(2,3) `CHECK_WGT(2,4) `CHECK_WGT(2,5) `CHECK_WGT(2,6) `CHECK_WGT(2,7)
        `CHECK_WGT(3,0) `CHECK_WGT(3,1) `CHECK_WGT(3,2) `CHECK_WGT(3,3) `CHECK_WGT(3,4) `CHECK_WGT(3,5) `CHECK_WGT(3,6) `CHECK_WGT(3,7)
        `CHECK_WGT(4,0) `CHECK_WGT(4,1) `CHECK_WGT(4,2) `CHECK_WGT(4,3) `CHECK_WGT(4,4) `CHECK_WGT(4,5) `CHECK_WGT(4,6) `CHECK_WGT(4,7)
        `CHECK_WGT(5,0) `CHECK_WGT(5,1) `CHECK_WGT(5,2) `CHECK_WGT(5,3) `CHECK_WGT(5,4) `CHECK_WGT(5,5) `CHECK_WGT(5,6) `CHECK_WGT(5,7)
        `CHECK_WGT(6,0) `CHECK_WGT(6,1) `CHECK_WGT(6,2) `CHECK_WGT(6,3) `CHECK_WGT(6,4) `CHECK_WGT(6,5) `CHECK_WGT(6,6) `CHECK_WGT(6,7)
        `CHECK_WGT(7,0) `CHECK_WGT(7,1) `CHECK_WGT(7,2) `CHECK_WGT(7,3) `CHECK_WGT(7,4) `CHECK_WGT(7,5) `CHECK_WGT(7,6) `CHECK_WGT(7,7)

        // update xin size
        issue_apb_tr(.addr(ADDR_input_xin_width_size ), .rewr(1'b1), .wdata(8-1), .rdate(rdata));
        issue_apb_tr(.addr(ADDR_input_xin_height_size), .rewr(1'b1), .wdata(8-1), .rdate(rdata));

        // update input/output acc size
        // acc_size = (xin_w * xin_h /4)
        issue_apb_tr(.addr(ADDR_input_acc_size       ), .rewr(1'b1), .wdata(8*8/4-1), .rdate(rdata));

        // update output channel acc start address
        for (int i = 0; i < (N_ACC_MX*N_ACC_PER_MX); i=i+2) begin
            logic [15:0] start_addr0;
            logic [15:0] start_addr1;
            start_addr0 =     i*8*8;
            start_addr1 = (i+1)*8*8;            
            issue_apb_tr(.addr(ADDR_output_acc_channel_start_addr + 4*i/2), .rewr(1'b1), .wdata({start_addr1, start_addr0}), .rdate(rdata));
        end
        
        // start systolic array
        issue_apb_tr(.addr(ADDR_start_sytolic_array), .rewr(1'b1), .wdata(32'b10), .rdate(rdata));

        //polling status
        do begin
            issue_apb_tr(.addr(ADDR_sytolic_array_idle), .rewr(1'b0), .wdata(32'bz), .rdate(rdata));
        end while(rdata[1] != 1'b1);

 
        
        // read the content from the deaccu memory to tell the correctness
        `READ_DEACC_RAM(0);
        $finish;
    end

endmodule


